import os
import sys
import numpy as np
import cv2

import torch
from tqdm import tqdm
# Logger
from logger.ConsoleLogger import MyColorConsoleLogger


    
from logger.TensorboardLogger import TensorBoardLogger as TensorBoardLogger

# Dataset
from dataset.frames_dataset_with_lmks import FramesDataset
from torch.utils.data import DataLoader

# Models
from modulesiris.generator import OcclusionAwareGenerator
from modulesiris.discriminator import Discriminator
import torch.optim as optim


# Loss
from modules.loss import GLoss, DLoss, detach_kp

#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DEBUG = True
if DEBUG:
    logger = MyColorConsoleLogger(process_name="Training", log_level="debug")
else:
    logger = MyColorConsoleLogger(process_name="Training", log_level="info")

# torch.autograd.set_detect_anomaly(True)
global_iteration_step = 0
device = 'cuda'

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logger.info('Save checkpoint to {0:}'.format(filename))

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img

def viz_prediction(x, x_prime, x_prime_hat, kp_driving, kp_src,epoch, batch, output_vis):
    """
    x: Bx3xHxW
    x_prime: Bx3xHxW
    x_prime_hat: Bx3xHxW

    """
    if not os.path.isdir(output_vis):
        os.makedirs(output_vis)
    _,_,h,w = x.shape
    
    x = x.detach().cpu().numpy()
    x_prime = x_prime.detach().cpu().numpy()
    x_prime_hat = x_prime_hat.detach().cpu().numpy()
    kp_src = kp_src.cpu().numpy()
    kp_driving = kp_driving.cpu().numpy()

    for i, (x1, x2, x3, ks, kd) in enumerate(zip(x, x_prime, x_prime_hat, kp_src, kp_driving)):
        x1 = (np.transpose(x1, (1,2,0))*255.0).astype(np.uint8)

        x2 = (np.transpose(x2, (1,2,0))*255.0).astype(np.uint8)

        x3 = (np.transpose(x3, (1,2,0))*255.0).astype(np.uint8)

        # ks = ks.squeeze(0)
        # kd = kd.squeeze(0)
        ks = (ks+1) * np.array([w,h]) / 2.0
        kd = (kd+1) * np.array([w,h]) / 2.0

        x1 = draw_landmarks(x1, ks)
        x2 = draw_landmarks(x2, kd)
        x3 = draw_landmarks(x3, kd)

        img = np.hstack((x1, x2, x3))
        cv2.imwrite(f'{output_vis}/epoch{epoch}_batch{batch}_sample{i}.png', img)


def track_model_norm_gradients(G, tensorboardLogger, global_iteration_step):
    total_norm = 0.0
    for p in D.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    tensorboardLogger.log(f'norm_grad_G', total_norm, global_iteration_step)


def track_model_gradients(model, tensorboardLogger, step, include_keywords=None, module_name="G"):
    for i, (tag, parm) in enumerate(model.named_parameters()):
        if include_keywords is not None:
            if not any([k in tag for k in include_keywords]):
                continue

        if hasattr(parm, 'grad'):
            tensorboardLogger.log_hist(tag=f"{module_name}_gradient_{tag}", value=parm.grad.data.cpu().numpy(), step=step)
            # logger.debug(f'grad of {tag}: {parm.grad.data.cpu().numpy()}')

        if hasattr(parm, 'data'):
            tensorboardLogger.log_hist(tag=f"{module_name}_weight_{tag}", value=parm.data.cpu().numpy(), step=step)
            # logger.debug(f'weights of {tag}: {parm.data.cpu().numpy()}')

def check_nan_grad(G, K, D):
    for name, param in D.named_parameters():
        if torch.isnan(param.grad).any().item() == True:
            logger.warning(f'There is Nan grad in D at param :{name}')
            return True
    for name, param in G.named_parameters():
        if torch.isnan(param.grad).any().item() == True:
            logger.warning(f'There is Nan grad in G at param :{name}')
            return True
    
    for name, param in K.named_parameters():
        if torch.isnan(param.grad).any().item() == True:
            logger.warning(f'There is Nan grad in K at param :{name}')
            return True
    return False

def log_grad_if_nan(G, K, D, kp_driving, kp_src):
    if check_nan_grad(G, K, D):
        import pdb; pdb.set_trace()
        # Save G
        torch.save(list(G.named_parameters()), "G_param.pt")
        grads = {}
        for name, param in G.named_parameters():
            grads[name] = param.grad
        torch.save(grads, "G_param_grad.pt")

        # Save D
        torch.save(list(D.named_parameters()), "D_param.pt")
        grads = {}
        for name, param in D.named_parameters():
            grads[name] = param.grad
        torch.save(grads, "D_param_grad.pt")

        # Save K
        torch.save(list(K.named_parameters()), "K_param.pt")
        grads = {}
        for name, param in K.named_parameters():
            grads[name] = param.grad
        torch.save(grads, "K_param_grad.pt")

        raise RuntimeError(f'Gradient is Nan')

def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return kp_appearance, kp_video

def train_one_epoch(G, D, K, dataloader, optimizerK, optimizerG, optimizerD, epoch, tensorBoardLogger, viz_output="./viz", loss_weight_dict=None):
    G.train()
    D.train()
    
    criterion_G = GLoss(loss_weight_dict=loss_weight_dict)
    criterion_D = DLoss(loss_weight_dict=loss_weight_dict)



    global global_iteration_step
    num_batch = len(dataloader)

    i = 0
    with tqdm(iter(dataloader),  unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batchdata in tepoch:
            i += 1

            if i > 1000:
                break

            global_iteration_step += 1
            x_prime, x = batchdata["driving"], batchdata["source"]
            kp_driving, kp_src = batchdata["lmks_driving"], batchdata["lmks_source"]
            # logger.info(kp_driving["value"].shape)
            # logger.info(kp_src["value"].shape)

            x_prime = x_prime.to(device)   
            x = x.to(device)    

            kp_driving["value"] = kp_driving["value"].to(device)
            kp_src["value"] = kp_src["value"].to(device)

            # Update G and K
            prediction = G(source_image=x, kp_driving=kp_driving, kp_source=kp_src)
            # logger.debug(f"prediction shape:{prediction['video_prediction'].shape}")
            loss_G_dict = criterion_G(D=D, x_prime=x_prime, x_hat_prime=prediction["prediction"], kp_driving=kp_driving)
            loss_G = loss_G_dict["total_loss_G"]
            loss_rec_G = loss_G_dict["loss_rec_G"]
            loss_adv_G = loss_G_dict["loss_adv_G"]
            loss_rec_vgg19_G = loss_G_dict["loss_rec_vgg19_G"]
            loss_G_K = loss_G 
            loss_G_K.backward()

            # track_model_gradients(G, tensorboardLogger, global_iteration_step, module_name="G")
            # track_model_gradients(D, tensorboardLogger, global_iteration_step, module_name="D")

            # track_model_norm_gradients(G, tensorboardLogger, global_iteration_step)
            # torch.nn.utils.clip_grad_norm_(list(D.parameters())+list(G.parameters())+list(K.parameters()), max_norm=5)

            # log_grad_if_nan(G, K, D, kp_driving, kp_src)
            
            # track_model_gradients(K, tensorboardLogger, global_iteration_step, include_keywords=["encoder.down_blocks.0"],module_name="K")
            # track_model_gradients(G, tensorboardLogger, global_iteration_step, include_keywords=["encoder.down_blocks.0"],module_name="G")
            
            optimizerG.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()


            # Update Discriminator
            kp_driving = detach_kp(kp_driving)
            kp_src = detach_kp(kp_src)
            loss_D_dict = criterion_D(D=D, x_prime=x_prime.detach(), x_hat_prime=prediction["prediction"].detach(), kp_driving=kp_driving)
            loss_D = loss_D_dict["loss_D"]
           
            loss_D.backward()
            optimizerD.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # # Log
            tepoch.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_reg_G=loss_rec_G.item(), loss_adv_G=loss_adv_G.item())
            # tepoch.set_postfix(loss_G=loss_G.item())

            tensorboardLogger.log(f"train/loss_g", loss_G.item(), epoch*num_batch+i)
            tensorboardLogger.log(f"train/loss_d", loss_D.item(), epoch*num_batch+i)
            tensorboardLogger.log(f"train/loss_rec_G", loss_rec_G.item(), epoch*num_batch+i)
            tensorboardLogger.log(f"train/loss_adv_G", loss_adv_G.item(), epoch*num_batch+i)
            tensorboardLogger.log(f"train/loss_rec_vgg19_G", loss_rec_vgg19_G.item(), epoch*num_batch+i)


            # logger.debug(f"Loss_g :{loss_G.item()}")
            if i<10:
                viz_prediction(x, x_prime, prediction["prediction"], kp_driving["value"], kp_src["value"], epoch, i, viz_output)
            tepoch.update(1)





def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d:
        print("Yoooooo")
        torch.nn.init.xavier_uniform(m.weight)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training first-order-motion')
    parser.add_argument('--exp_name', default="baseline", type=str)
    parser.add_argument('--viz_output', default="./viz", type=str)
    parser.add_argument('--root_log', default="./logs", type=str)
    parser.add_argument('--lamda_rec', default=10.0, type=float)
    parser.add_argument('--lamda_adv', default=1.0, type=float)
    parser.add_argument('--lamda_rec_vgg', default=10.0, type=float)
    parser.add_argument('--lamda_equi_value_loss', default=10.0, type=float)
    parser.add_argument('--lamda_equi_jacobian_loss', default=10.0, type=float)
    parser.add_argument('--using_first_order_motion', default=1, type=int)
    parser.add_argument('--num_kp', default=10, type=int)


    args = parser.parse_args()

    loss_weight_dict = {"lamda_rec":args.lamda_rec, "lamda_adv":args.lamda_adv, "lamda_rec_vgg":args.lamda_rec_vgg,\
        "lamda_equi_value_loss":args.lamda_equi_value_loss, "lamda_equi_jacobian_loss":args.lamda_equi_jacobian_loss, "using_first_order_motion":args.using_first_order_motion}



    
  

    ################## Config for Vox
    # Dataset and dataloader
    # root_dir = "../MonkeyNet/data/vox"
    root_dir = "./data/eth_motion_data"

    augmentation_params = {"flip_param" : {"horizontal_flip": False, "time_flip":False}, "jitter_param" :{"brightness":0.1, "contrast":0.1, "saturation":0.1, "hue":0.1}}
    # augmentation_params = {"flip_param" : {"horizontal_flip": True, "time_flip":True}, "jitter_param" :{"brightness":0.1, "contrast":0.1, "saturation":0.1, "hue":0.1}}
    dataset = FramesDataset(root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=augmentation_params)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=8, drop_last=True)
    logger.info(f'Len dataloader :{len(dataloader)}')
    # Model here
    dense_motion_params = {"block_expansion":64, "max_features": 1024, "num_blocks":5, "scale_factor":0.25}
    G = OcclusionAwareGenerator(num_channels=3, num_kp=args.num_kp, block_expansion=64, max_features=512, num_down_blocks=2,
                 num_bottleneck_blocks=6, estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=args.using_first_order_motion)
    
    D = Discriminator(num_channels=3, block_expansion=32, num_blocks=4, max_features=512,
                sn=True, use_kp=False, num_kp=args.num_kp, kp_variance=0.01, estimate_jacobian= args.using_first_order_motion)
    
    G.to(device)
    D.to(device)

    # K.module.apply(init_weights)
    # G.module.apply(init_weights)
    # D.module.apply(init_weights)


    # Optimizer and Scheduler here
    optimizerG = optim.Adam(params =  list(G.parameters()),
                            lr=2e-4,
                            amsgrad=False)

    optimizerD = optim.Adam(params = list(D.parameters()),
                        lr=2e-4,
                        amsgrad=False)

    MAX_EPOCH = 1000
    experiment_name=args.exp_name

    log = args.root_log
    if not os.path.isdir(f"checkpoints/{experiment_name}"):
        os.makedirs(f"checkpoints/{experiment_name}")
    
    if not os.path.isdir(f"{log}/{experiment_name}"):
        os.makedirs(f"{log}/{experiment_name}")

    tensorboardLogger = TensorBoardLogger(root=log, experiment_name=experiment_name)
    K = None
    optimizerK = None
    for epoch in range(0, MAX_EPOCH):
        train_one_epoch(G, D, K, dataloader, optimizerK, optimizerG, optimizerD, epoch, tensorboardLogger, args.viz_output, loss_weight_dict)      
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints/{experiment_name}/{epoch+1}.pth.tar'
            save_checkpoint({
                    'epoch': epoch+1,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, filename=f'{checkpoint_path}')