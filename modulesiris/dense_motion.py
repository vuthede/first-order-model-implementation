from torch import nn
import torch.nn.functional as F
import torch
from modulesiris.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


# class JacobianEstimation(nn.Module):
#     def __init__(self,block_expansion=64, num_blocks=5, max_features=1024, num_kp=10, num_channels=3,kp_variance=0.01):
#         super(JacobianEstimation, self).__init__()
#         self.num_kp = num_kp
#         self.num_channels = num_channels
#         self.kp_variance = kp_variance
#         self.hourglass = Hourglass(block_expansion=block_expansion, in_features=3+num_kp+1,
#                                    max_features=max_features, num_blocks=num_blocks)
#         self.jacobiannet = nn.Conv2d(self.hourglass.out_filters, 4*num_kp , kernel_size=(7, 7), padding=(3, 3))
#         self.jacobiannet_d = nn.Conv2d(self.hourglass.out_filters, 4*num_kp , kernel_size=(7, 7), padding=(3, 3))

        
#         self.jacobiannet.weight.data.zero_()
#         self.jacobiannet.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_kp, dtype=torch.float))
#         self.jacobiannet_d.weight.data.zero_()
#         self.jacobiannet_d.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_kp, dtype=torch.float))
#         # self.tan = nn.Tanh()

#     def forward(self, src, kp_driving, kp_src):
#         spatial_size = src.shape[2:]
#         h_d = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
#         h_s = kp2gaussian(kp_src, spatial_size=spatial_size, kp_variance=self.kp_variance)
#         heatmap = h_d - h_s
#         # heatmap = h_d


#         # heatmap = h_s
#         #adding background feature
#         zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
#         heatmap1 = torch.cat([src, zeros, heatmap], dim=1) # Bx(3+K+1)x256x256
#         out = self.hourglass(heatmap1)
        

#         # Src
#         jacobian = self.jacobiannet(out)
#         b,c,h,w = src.shape
#         jacobian = jacobian.reshape(b, self.num_kp, 4, h, w)
#         h_s = h_s.unsqueeze(2) # bx1x1x256x256
#         jacobian = jacobian * h_s # Bxnum_kpx4x256x256
#         jacobian = jacobian.view(kp_driving["value"].shape[0], kp_driving["value"].shape[1], 4, -1) 
#         jacobian = jacobian.sum(dim=-1) # Bxnum_kpx4
#         jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) # Bx10x2x2

#         # Driving
#         jacobian_d = self.jacobiannet_d(out)
#         b,c,h,w = src.shape
#         jacobian_d = jacobian_d.reshape(b, self.num_kp, 4, h, w)
#         h_d = h_d.unsqueeze(2) # bx1x1x256x256
#         jacobian_d = jacobian_d * h_d # Bxnum_kpx4x256x256
#         jacobian_d = jacobian_d.view(kp_driving["value"].shape[0], kp_driving["value"].shape[1], 4, -1) 
#         jacobian_d = jacobian_d.sum(dim=-1) # Bxnum_kpx4
#         jacobian_d = jacobian_d.view(jacobian_d.shape[0], jacobian_d.shape[1], 2, 2) # Bx10x2x2


#         return jacobian, jacobian_d


class JacobianEstimation(nn.Module):
    def __init__(self,block_expansion=64, num_blocks=5, max_features=1024, num_kp=10, num_channels=3,kp_variance=0.01):
        super(JacobianEstimation, self).__init__()
        self.num_kp = num_kp
        self.num_channels = num_channels
        self.kp_variance = kp_variance
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=3+num_kp+1,
                                   max_features=max_features, num_blocks=num_blocks)
        self.jacobiannet = nn.Conv2d(self.hourglass.out_filters, 4*num_kp , kernel_size=(7, 7), padding=(3, 3))
        self.jacobiannet_d = nn.Conv2d(self.hourglass.out_filters, 4*num_kp , kernel_size=(7, 7), padding=(3, 3))

        
        self.jacobiannet.weight.data.zero_()
        self.jacobiannet.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_kp, dtype=torch.float))
        self.jacobiannet_d.weight.data.zero_()
        self.jacobiannet_d.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_kp, dtype=torch.float))

    def forward(self, src, kp_driving, kp_src):
        spatial_size = src.shape[2:]
        h_d = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        h_s = kp2gaussian(kp_src, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = h_d - h_s

        # heatmap = h_s
        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap1 = torch.cat([src, zeros, heatmap], dim=1) # Bx(3+K+1)x256x256
        out = self.hourglass(heatmap1)
        
        # Original 
        jacobian = self.jacobiannet(out)
        b,c,h,w = src.shape
        jacobian = jacobian.reshape(b, self.num_kp, 4, h, w)
        heatmap = heatmap.unsqueeze(2) # bx1x1x256x256
        jacobian = jacobian * heatmap # Bxnum_kpx4x256x256
        jacobian = jacobian.view(kp_driving["value"].shape[0], kp_driving["value"].shape[1], 4, -1) 
        jacobian = jacobian.sum(dim=-1) # Bxnum_kpx4
        jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2) # Bx10x2x2

        return jacobian

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class RefineEyelidMotion(nn.Module):
    def __init__(self,block_expansion=64, num_blocks=5, max_features=1024, num_kp=10, num_channels=3,kp_variance=0.01):
        super(RefineEyelidMotion, self).__init__()
        self.num_kp = num_kp
        self.num_channels = num_channels
        self.kp_variance = kp_variance
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=3+2,
                                   max_features=max_features, num_blocks=num_blocks)
        self.refine_layer = nn.Conv2d(self.hourglass.out_filters, 2 , kernel_size=(7, 7), padding=(3, 3))
        self.combine_layer = nn.Conv2d(2, 2 , kernel_size=(3, 3), padding=(1, 1))
        self.device = 'cuda'

    def forward(self, src, kp_driving, kp_src, sparse_motion):
        spatial_size = src.shape[2:]
        h_d = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance, key='value_witheyelid')
        h_s = kp2gaussian(kp_src, spatial_size=spatial_size, kp_variance=self.kp_variance, key='value_witheyelid')
        # heatmap = h_d - h_s # bsxKxwxh
        heatmap = h_d

        heatmap = heatmap.max(dim=1, keepdims=False)[0] # bsxwxh
        heatmap = heatmap.unsqueeze(1) # bsx1xwxh
        heatmap = heatmap.to(self.device)
        # import pdb; pdb.set_trace()
        # heatmap = h_s
        #adding background feature
        # zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        # import pdb; pdb.set_trace();
        heatmap1 = torch.cat([src, sparse_motion.squeeze(1).permute(0,3,1,2)], dim=1) # Bx(3+2)x256x256
        out = self.hourglass(heatmap1)
        
        # Original 
        replacement_lidmotion = self.refine_layer(out)
        b,c,h,w = src.shape
        replacement_lidmotion = replacement_lidmotion.reshape(b, 2, h, w)
        # replacement_lidmotion = replacement_lidmotion * heatmap # Bx2x256x256
        
        # Concat sparse_motion + replacementlidmotion
        # sparse_motion: #bsx1xhxwx2
        input_refine = sparse_motion.squeeze(1).permute(0,3,1,2) + replacement_lidmotion # Bx2x256xx256
        sparse_motion_refine = self.combine_layer(input_refine) # Bx2x256x256
        sparse_motion_refine = sparse_motion_refine.permute(0,2,3,1).unsqueeze(1) #bsx1xhxwx2
        return sparse_motion_refine

class TPS:
    '''
    / Ref from https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.
    '''
    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == 'random':
            noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0, 
                        std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        elif mode == 'kp':
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:,:,:, None]-kp_1[:,:, None, :], dim=4, p=2)
            K = K**2
            K = K * torch.log(K+1e-9)
#             print(f'K:{K}')
            
            one1 = torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1).to(device).type(kp_type)
            kp_1p = torch.cat([kp_1,one1], 3)
            
            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero],2)
            L = torch.cat([K,kp_1p.permute(0,1,3,2)],2)
            L = torch.cat([L,P],3)
        
            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            Y = torch.cat([kp_2, zero], 2)
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type)*0.01
            L = L + one

            param = torch.matmul(torch.inverse(L),Y)
            self.theta = param[:,:,n:,:].permute(0,1,3,2)
#             print(f'Theta :{self.theta}')

            self.control_points = kp_1
            self.control_params = param[:,:,:n,:]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == 'kp':
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(coordinates.device)

        if self.mode == 'kp':
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]

            distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(self.bs, control_points.shape[1], -1, 1, 2)

            distances = distances ** 2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == 'random':
            theta = theta.unsqueeze(1)
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = ances ** 2

            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise Exception("Error TPS mode")

        return transformed


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01, using_first_order_motion=True, using_thin_plate_spline_motion=False, estimate_lid_motion=False):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        self.jacobian_estimator = JacobianEstimation(block_expansion=block_expansion, num_blocks=num_blocks, max_features=max_features, num_kp=num_kp, num_channels=num_channels,kp_variance=kp_variance)

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.using_first_order_motion = using_first_order_motion
        self.using_thin_plate_spline_motion = using_thin_plate_spline_motion
        # print(f'YOOOOOOOOOOO using_thin_plate_spline_motion :{using_thin_plate_spline_motion}')
        # print(f'YOOOOOOOOOOO using_first_order_motion :{using_first_order_motion}')
        self.device = 'cuda'

        self.refine_eyelid_motion_module = RefineEyelidMotion(block_expansion=64, num_blocks=5, max_features=1024, num_kp=10, num_channels=3,kp_variance=0.01)
        self.estimate_lid_motion = estimate_lid_motion

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap
    
    def create_sparse_motion_given_kps(self, source_image, kp_driving_value, kp_src_value):
        bs, _, h, w = source_image.shape
        
          # Control points for driving
        # kp_1 = kp_driving['value'].unsqueeze(1) # bsx1xKx2
        # fix_kp1 = torch.FloatTensor([[[[-1,-1],[-1,1], [1,-1], [1,1]]]]) #1x1x4x2
        # fix_kp1 = fix_kp1.repeat(bs, 1, 1, 1) #bsx1x4x2
        # fix_kp1 = fix_kp1.to(self.device)
        # kp_1 = torch.cat([fix_kp1, kp_1], dim=2) #bsx1x(4+K)x2
        
        # Control points for source
        # kp_2 = kp_src['value'].unsqueeze(1) # bsx1xKx2
        # fix_kp2 = torch.FloatTensor([[[[-1,-1],[-1,1], [1,-1], [1,1]]]]) #1x1x4x2
        # fix_kp2 = fix_kp2.repeat(bs, 1, 1, 1) #bsx1x4x2
        # fix_kp2 = fix_kp2.to(self.device)
        # kp_2 = torch.cat([fix_kp2, kp_2], dim=2) #bsx1x(4+K)x2

        # Control points for driving
        kp_1 = kp_driving_value.unsqueeze(1) # bsx1xKx2
        kp_2 = kp_src_value.unsqueeze(1) # bsx1xKx2
        # Thin plate spline motion
        # Drivinng should have shape Bxnum_setxKx2, where in our cases num_set=1
        # Beside that, we should append some edge anchors to prevent the image is distorted alot
        transform = TPS(mode='kp', bs=bs,sigma_affine=0.05, sigma_tps=0.005, kp_1=kp_1, kp_2=kp_2, points_tps=None)
        driving_to_source = transform.transform_frame(source_image) #bsx1xhxwx2
        
        return driving_to_source


    def create_sparse_thin_plate_spline_motion(self, source_image, kp_driving, kp_src, estimate_lid_motion=False):
        # print(f'Using thin plate ##############################################3!')
        # Identity grid
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_src['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        
        # Get thin plate motion
        driving_to_source = self.create_sparse_motion_given_kps(source_image, kp_driving["value"], kp_src["value"])  
        driving_to_source_with_eyelid_motion_groundtruth = self.create_sparse_motion_given_kps(source_image, kp_driving["value_witheyelid"], kp_src["value_witheyelid"])  

        if estimate_lid_motion:
            driving_to_source_with_eyelid_motion_prediction = self.refine_eyelid_motion_module(source_image, kp_driving=kp_driving, kp_src=kp_src, sparse_motion=driving_to_source)
            driving_to_source = driving_to_source_with_eyelid_motion_prediction
        else:
            driving_to_source_with_eyelid_motion_prediction = None

        K = kp_driving['value'].shape[-2]
        driving_to_source = driving_to_source.repeat(1, K, 1, 1, 1) #bsxKxhxwx2

        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1) #bsx(K+1)xhxwx2

        return {"sparse_motion":sparse_motions, "driving_to_source_with_eyelid_motion_groundtruth": driving_to_source_with_eyelid_motion_groundtruth, "driving_to_source_with_eyelid_motion_prediction":driving_to_source_with_eyelid_motion_prediction}

    def create_sparse_motions(self, source_image, kp_driving, kp_source, jacobianD2S, using_first_order_motion=True):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)

        # z- zk
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        
        # TS<--D(z) = TS<--D(zk) + jacobS<--D * (z - zk)
        if using_first_order_motion:
            jacobianD2S = jacobianD2S.unsqueeze(-3).unsqueeze(-3)
            jacobianD2S = jacobianD2S.repeat(1, 1, h, w, 1, 1)        
            coordinate_grid =  torch.matmul(jacobianD2S, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = kp_source['value'].view(bs, self.num_kp, 1, 1, 2) + coordinate_grid

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions


        # bs, _, h, w = source_image.shape
        # identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        # identity_grid = identity_grid.view(1, 1, h, w, 2)
        # coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        
        # # TS<--D(z) = TS<--D(zk) + jacobS<--D * (z - zk)
        # if using_first_order_motion:
        #     jacobian = torch.matmul(jacobianD2S, torch.inverse(jacobianD2S_D))
        #     jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
        #     jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
        #     coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
        #     coordinate_grid = coordinate_grid.squeeze(-1)

        # driving_to_source = kp_source['value'].view(bs, self.num_kp, 1, 1, 2) + coordinate_grid

        # #adding background feature
        # identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        # sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        # return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape
        out_dict = dict()
        # 1.Create heatmap representation
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        out_dict['heatmap_representation'] = heatmap_representation
        
        if self.using_first_order_motion:
            # 2. Estimate jacobian
            # jacobianD2S, jacobianD2S_D = self.jacobian_estimator(source_image, kp_driving, kp_source)
            jacobianD2S = self.jacobian_estimator(source_image, kp_driving, kp_source)
            out_dict['jacobianD2S'] = jacobianD2S
            # 3. Estimate motion
            # sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source, jacobianD2S, jacobianD2S_D,using_first_order_motion=self.using_first_order_motion)
            sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source, jacobianD2S,using_first_order_motion=self.using_first_order_motion)

        elif self.using_thin_plate_spline_motion:
            motion_dict = self.create_sparse_thin_plate_spline_motion(source_image, kp_driving, kp_source, estimate_lid_motion=self.estimate_lid_motion)
            sparse_motion = motion_dict["sparse_motion"]
        
        out_dict['sparse_motion'] = sparse_motion
        out_dict['driving_to_source_with_eyelid_motion_groundtruth'] = motion_dict['driving_to_source_with_eyelid_motion_groundtruth']
        out_dict['driving_to_source_with_eyelid_motion_prediction'] = motion_dict['driving_to_source_with_eyelid_motion_prediction']

        
        #3. Deformed images
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict



if __name__ == '__main__':

    B=1
    K=1
    src = torch.rand(B, 3, 256, 256)
    kp_driving = {"value": torch.rand(B, K, 2)}
    kp_src = {"value": torch.rand(B, K, 2)}
    # jacob_estimator =  JacobianEstimation(block_expansion=64, num_blocks=5, max_features=1024, num_kp=K, num_channels=3,kp_variance=0.01)
    # jaco = jacob_estimator(src, kp_driving, kp_src)
    # import pdb;pdb.set_trace()

    dense_motion = DenseMotionNetwork(block_expansion=64, num_blocks=5, max_features=1024, num_kp=K, num_channels=3, estimate_occlusion_map=True,scale_factor=1, kp_variance=0.01)
    
    out = dense_motion(src, kp_driving, kp_src)
    # out["sparse_deformed"] # Bx11x3x256x256. Deformed of source images. Those together with heatmaps will be pushed into a network to generate the masks, where motion is occur
    # out["mask"] # Bx11x256x256. The mask 
    # ** out["deformation"]. # Bx256x256x2. The final deformation after apply sparse_motion * mask
    # ** out["occlusion_map"] # Bx1x256x256. The occlusion map
    # ** Note: In the generator network, the deformation will be used to deform the source image/source image featur map
    #          After warping feature maps, those will be multiplied to the occlusion map to mask out occlusion region, it is like the area should be inpainted after warping
    import pdb;pdb.set_trace()

    print("End")