"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modulesiris.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad

def mean_batch(val):
    return val.view(val.shape[0], -1).mean(-1)

## Generator loss. They will include Kp_detector also.
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False, device='cuda'):
        super(Vgg19, self).__init__()
        vgg19 = models.vgg19(pretrained=True).to(device)
        vgg_pretrained_features = vgg19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        # self.mean = self.mean.to(device)
        # self.std = self.std.to(device)


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class AdvLossG(nn.Module):
    def __init__(self):
        super(AdvLossG, self).__init__()
        self.scale_factor = 0.25


    def forward(self, D, x_hat_prime, kp_driving):
        maps, score = D(x_hat_prime, kp_driving)
        loss = mean_batch((score - 1)*(score - 1))
        loss = torch.mean(loss)

        return loss        

class RecLossG(nn.Module):
    def __init__(self):
        super(RecLossG, self).__init__()
        self.scale_factor = 0.25


    def forward(self, D, x_prime, x_hat_prime, kp_driving):
        # Predict
        maps1, _ = D(x_hat_prime, kp_driving)

        # GT
        maps2, _ = D(x_prime, kp_driving)

        loss = 0.0
        for fea_pred, fea_gt in zip(maps1[:-1], maps2[:-1]):
            loss = loss + torch.mean(mean_batch(torch.abs(fea_pred - fea_gt)))

        return loss

class RecVGGLossG(nn.Module):
    def __init__(self, device='cuda'):
        super(RecVGGLossG, self).__init__()
        self.scale_factor = 0.25
        self.device = device
        self.vgg = Vgg19()
        self.vgg.to(self.device)
        self.vgg.eval()


    def forward(self, x_prime, x_hat_prime):
        # Predict
        maps1 = self.vgg(x_hat_prime)

        # GT
        maps2 = self.vgg(x_prime)

        loss = 0.0
        for fea_pred, fea_gt in zip(maps1[:-1], maps2[:-1]):
            loss = loss + torch.mean(mean_batch(torch.abs(fea_pred - fea_gt)))

        return loss

class GLoss(nn.Module):
    def __init__(self, loss_weight_dict):
        super(GLoss, self).__init__()
        self.loss_adv_G = AdvLossG()
        self.loss_rec_G = RecLossG()
        self.loss_vgg19_G = RecVGGLossG()
        self.lamda_rec = loss_weight_dict["lamda_rec"]
        self.lamda_adv = loss_weight_dict["lamda_adv"]
        self.lamda_rec_vgg = loss_weight_dict["lamda_rec_vgg"]

    def forward(self, D, x_prime, x_hat_prime,  kp_driving):
        loss_G = self.loss_adv_G(D, x_hat_prime, kp_driving)
        loss_rec_G = self.loss_rec_G(D, x_prime, x_hat_prime, kp_driving)
        loss_rec_vgg = self.loss_vgg19_G(x_prime, x_hat_prime)
        total_loss = self.lamda_rec * loss_rec_G + self.lamda_adv * loss_G + self.lamda_rec_vgg * loss_rec_vgg
        return {"total_loss_G" : total_loss, "loss_rec_G":loss_rec_G, "loss_adv_G":loss_G, "loss_rec_vgg19_G":loss_rec_vgg}

## Disciminator loss
class DLoss(nn.Module):
    def __init__(self, loss_weight_dict):
        super(DLoss, self).__init__()
        self.scale_factor = 0.25
    
    def forward(self, D, x_prime, x_hat_prime, kp_driving):
        # # Predict
        maps1, score_fake = D(x_hat_prime, kp_driving)
        loss_fake = score_fake*score_fake

        # GT
        maps2, score_real = D(x_prime,kp_driving)
        loss_real = (score_real - 1)*(score_real-1)
        
        loss = mean_batch(loss_fake + loss_real)
        loss = torch.mean(loss)

        return {"loss_D": loss}


## Equivariance loss to constraint Kp detector
def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}
class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        # print(kwargs)
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)

        # # Invert grid
        # identity = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0) # 1xhxwx2
        # delta = grid - identity
        # optical_flow = identity - delta

        return F.grid_sample(frame, grid, padding_mode="reflection")
        # return F.grid_sample(frame, optical_flow, padding_mode="reflection")


    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


class KEquivarianceLoss(nn.Module):
    def __init__(self,loss_weight_dict, device= 'cuda'):
        super(KEquivarianceLoss, self).__init__()
        self.lamda_equi_value_loss = loss_weight_dict["lamda_equi_value_loss"]
        self.lamda_equi_jacobian_loss = loss_weight_dict["lamda_equi_jacobian_loss"]
        self.using_first_order_motion = loss_weight_dict["using_first_order_motion"]
        self.device = device
        self.down = AntiAliasInterpolation2d(channels=3, scale=0.25)
        self.down.to(self.device)

    # def forward(self, K, x_origin, x_prime, kp_src, kp_driving):
    #     transform = Transform(x_prime.shape[0], sigma_affine=0.05, sigma_tps=0.005, points_tps=5)
    #     # transformed_frame = transform.transform_frame(x_prime) # New warped image
    #     transformed_kp = {"value":transform.warp_coordinates(kp_driving["value"])}

    #     x_origin_down = self.down(x_origin)
    #     # x_prime_down = self.down(x_prime)

    #     TS_Y = K(src=x_origin_down, kp_driving=transformed_kp, kp_src=kp_src) 
    #     TS_D = K(src=x_origin_down, kp_driving=kp_driving, kp_src=kp_src) 
    #     loss_equivariance_value = torch.tensor(0.0, device=self.device,requires_grad=True)
    #     ## Loss equivariance_jacobian
    #     # Equation (13) on paper: TX<--Y * TY<=--R
    #     if self.using_first_order_motion:
    #         kp_driving['value'].requires_grad = True
    #         TY_D = transform.jacobian(kp_driving['value'])
                                                        
    #         # (TX<--R)^-1 TX<--Y * TY<--R: Original paper
    #         # We change a bit like
    #         # (TD<--S)^-1 TD<--Y * TY<--S: We propose
    #         # I = (TS<--Y)^-1  * TS<-D * TD<--Y
    #         tmp = torch.matmul(TS_D, torch.inverse(TY_D))
    #         # print(TS_Y)
    #         value = torch.matmul(torch.inverse(TS_Y), tmp) 
    #         eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
    #         loss_equivariance_jacobian = torch.abs(eye - value).mean()
    #     else:
    #         loss_equivariance_jacobian = torch.tensor(0.0, device=self.device,requires_grad=True)
    #     loss = self.lamda_equi_value_loss * loss_equivariance_value + self.lamda_equi_jacobian_loss * loss_equivariance_jacobian

    #     return {"loss_equivariance_K": loss, "loss_equivariance_value":loss_equivariance_value, "loss_equivariance_jacobian":loss_equivariance_jacobian}

    def forward(self, K, x_origin, x_prime, kp_src, kp_driving):
        x_origin_down = self.down(x_origin)
        x_prime_down = self.down(x_prime)

        TS_D = K(src=x_origin_down, kp_driving=kp_driving, kp_src=kp_src) 
        TD_S = K(src=x_prime_down, kp_driving=kp_src, kp_src=kp_driving) 
        loss_equivariance_value = torch.tensor(0.0, device=self.device,requires_grad=True)
        ## Loss equivariance_jacobian
        # Equation (13) on paper: TX<--Y * TY<=--R
        if self.using_first_order_motion:
            # constraint I = TS_D * TD_S
            value = torch.matmul(TS_D, TD_S) 
            eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
            loss_equivariance_jacobian = torch.abs(eye - value).mean()
        else:
            loss_equivariance_jacobian = torch.tensor(0.0, device=self.device,requires_grad=True)
        loss = self.lamda_equi_value_loss * loss_equivariance_value + self.lamda_equi_jacobian_loss * loss_equivariance_jacobian

        return {"loss_equivariance_K": loss, "loss_equivariance_value":loss_equivariance_value, "loss_equivariance_jacobian":loss_equivariance_jacobian}


if __name__ == "__main__":
    from discriminator import Discriminator 
    from generator import OcclusionAwareGenerator 
    from keypoint_detector import KPDetector 

    device = 'cuda'
    # Init model
    K = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1,
                 estimate_jacobian=True, scale_factor=1, single_jacobian_map=False, pad=0)
    D = Discriminator(num_channels=3, block_expansion=32, num_blocks=4, max_features=512,
                sn=True, use_kp=False, num_kp=10, kp_variance=0.01, estimate_jacobian= True)
    
    dense_motion_params = {"block_expansion":64, "max_features": 1024, "num_blocks":5, "scale_factor":0.25}
    G = OcclusionAwareGenerator(num_channels=3, num_kp=10, block_expansion=64, max_features=512, num_down_blocks=2,
                 num_bottleneck_blocks=6, estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=False)

    K.to(device)
    D.to(device)
    G.to(device)


    # Data fake
    B = 1
    src = torch.rand(B, 3, 256, 256).to(device)
    driving = torch.rand(B, 3, 256, 256).to(device)

    # Gen keypoints
    kp_driving = K(driving)
    kp_src = K(src)

    # Gen the image
    prediction = G(src, kp_driving, kp_src)
    x_hat_prime = prediction["prediction"]

    # Loss
    loss_G_criterion = GLoss() 
    loss_G_dict = loss_G_criterion(D, x_prime=driving, x_hat_prime=x_hat_prime,  kp_driving=kp_driving)
    for k, v in loss_G_dict.items():
        print(f'{k}: {v.item()}')
    
    # Loss to constraint keypoint
    loss_K_criterion = KEquivarianceLoss()
    loss_K_dict = loss_K_criterion(K, driving, kp_driving)
    for k, v in loss_K_dict.items():
        print(f'{k}: {v.item()}')

    # Loss dicriminator
    loss_D_criterion = DLoss()
    loss_D_dict = loss_D_criterion(D, driving, x_hat_prime, kp_driving)
    for k, v in loss_D_dict.items():
        print(f'{k}: {v.item()}')

    
