import os
import sys
import numpy as np
import cv2

import torch
from tqdm import tqdm

# Dataset
from dataset.frames_dataset_with_lmks import FramesDataset
from torch.utils.data import DataLoader

# Models
from modulesiris.generator import OcclusionAwareGenerator

# Loss
device = 'cuda'

def load_model(ckpt):
    checkpoint = torch.load(ckpt, map_location=device)
     # Model here
    dense_motion_params = {"block_expansion":64, "max_features": 1024, "num_blocks":5, "scale_factor":0.25}
    G = OcclusionAwareGenerator(num_channels=3, num_kp=2, block_expansion=64, max_features=512, num_down_blocks=2,
                 num_bottleneck_blocks=6, estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=True)
    
    G.load_state_dict(checkpoint["G_state_dict"], strict=True)
    G = G.to(device)
    return G

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 3, color, -1, lineType=cv2.LINE_AA)

    return img

def visualize(x, x_prime, x_prime_hat, x_prime_hat1, kp_src, kp_driving, output_vis="./debug", batch=0):
        """
        x: Bx3x1xHxW
        x_prime: Bx3x1xHxW
        x_prime_hat: Bx3x1xHxW
        kp_src: Bx1x10x2
        kp_driving: Bx1x10x2
        """
        if not os.path.isdir(output_vis):
            os.makedirs(output_vis)
        _,_,_,h,w = x.shape
        x = x.detach().cpu().numpy()
        x_prime = x_prime.detach().cpu().numpy()
        x_prime_hat = x_prime_hat.detach().cpu().numpy()
        x_prime_hat1 = x_prime_hat1.detach().cpu().numpy()


        kp_src = kp_src.detach().cpu().numpy()
        kp_driving = kp_driving.detach().cpu().numpy()

        for i, (x1, x2, x3,x4, ks, kd) in enumerate(zip(x, x_prime, x_prime_hat, x_prime_hat1, kp_src, kp_driving)):
            x1 = (np.transpose(x1, (2,3,0,1))*255.0).astype(np.uint8)
            x1 = x1.squeeze(-1)

            x2 = (np.transpose(x2, (2,3,0,1))*255.0).astype(np.uint8)
            x2 = x2.squeeze(-1)

            x3 = (np.transpose(x3, (2,3,0,1))*255.0).astype(np.uint8)
            x3 = x3.squeeze(-1)

            # import pdb; pdb.set_trace()
            # x4 = (np.transpose(x4, (2,3,0,1))*255.0).astype(np.uint8)
            # x4 = x4.squeeze(-1)
            x4 = (x4.squeeze(0)*255.0).astype(np.uint8)

            ks = ks.squeeze(0)
            kd = kd.squeeze(0)
            
            ks = (ks+1) * np.array([w,h]) / 2.0
            kd = (kd+1) * np.array([w,h]) / 2.0
            # import pdb; pdb.set_trace();
            # print(ks)
            # print(kd)

            x1 = draw_landmarks(x1, ks)
            x2 = draw_landmarks(x2, kd)
            x3 = draw_landmarks(x3, kd)
            # x4 = draw_landmarks(x4, kd)
            x4 = cv2.resize(x4, (x1.shape[1],x1.shape[0]))
            img = np.hstack((x1, x2, x3, x4))
            cv2.imwrite(f'{output_vis}/batch{batch}_sample{i}.png', img)


def vis(x, x_prime_hat, kp_src, kp_driving):
        """
        x: Bx3x1xHxW
        x_prime: Bx3x1xHxW
        x_prime_hat: Bx3x1xHxW
        kp_src: Bx1x10x2
        kp_driving: Bx1x10x2
        """
        _,_,h,w = x.shape
        x = x.detach().cpu().numpy()
        x_prime_hat = x_prime_hat.detach().cpu().numpy()


        kp_src = kp_src.detach().cpu().numpy()
        kp_driving = kp_driving.detach().cpu().numpy()

        for i, (x1, x3, ks, kd) in enumerate(zip(x, x_prime_hat, kp_src, kp_driving)):
            x1 = (np.transpose(x1, (1,2,0))*255.0).astype(np.uint8)
            x3 = (np.transpose(x3, (1,2,0))*255.0).astype(np.uint8)
            ks = (ks+1) * np.array([w,h]) / 2.0
            kd = (kd+1) * np.array([w,h]) / 2.0
            x1 = draw_landmarks(x1, ks)
            x3 = draw_landmarks(x3, ks)
            x3 = draw_landmarks(x3, kd, color=(0,255,255))

            img = np.hstack((x1, x3))
            return img



def synthize_kp_driving(kp_src):
    kp_driving = {}
    kp_driving["value"] =  kp_src["value"].clone()
    kp_driving["value"][:,:,0] = kp_driving["value"][:,:,0] +  np.random.uniform(-0.15, 0.15)
    kp_driving["value"][:,:,1] = kp_driving["value"][:,:,1] +  np.random.uniform(-0.03, 0.03)
    return kp_driving


if __name__ == '__main__':
    from skimage import io, img_as_float32

    # Load model
    ckpt = "checkpoints/motion_iris/21.pth.tar"
    G = load_model(ckpt = ckpt)
    G.eval()

    # Dataset
    root_dir = "./data/eth_motion_data"
    augmentation_params = {"flip_param" : {"horizontal_flip": False, "time_flip":False}, "jitter_param" :{"brightness":0.1, "contrast":0.1, "saturation":0.1, "hue":0.1}}
    dataset = FramesDataset(root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=augmentation_params)


    out = cv2.VideoWriter(f'motion_iris.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256*2, 256))

    for index in range(0,1):
        # batchdata = dataset[index]
        # _, x = batchdata["driving"], batchdata["source"]
        # _, kp_src = batchdata["lmks_driving"], batchdata["lmks_source"]

        # Fake image
        src_path = "./trinh.png"
        src = cv2.imread(src_path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (256, 256)) #BxCxDxH,W
        src = src/255.0
        x  = np.transpose(src, (2,0,1)) # 3x256x256
        kp_src = {"value": torch.FloatTensor([[-0.2136181,-0.31389177],[0.373667,-0.22871798]])}



        x = torch.FloatTensor(x)
        kp_src["value"] = torch.FloatTensor(kp_src["value"])


        x = x.to(device) 
        kp_src["value"] = kp_src["value"].to(device)
        

        kp_src["value"].unsqueeze_(0) 
        x.unsqueeze_(0) 

        

        import imageio
        img_list = []
        for i in range(100):
            kp_driving = synthize_kp_driving(kp_src)
            kp_driving["value"] = kp_driving["value"].to(device)

            prediction = G(source_image=x, kp_driving=kp_driving, kp_source=kp_src)

            img_out = vis(x, prediction["prediction"], kp_src["value"], kp_driving["value"])
            img_list.append(img_out)
        
            print(img_out.shape)
            # out.write(img_out)
        
        imageio.mimsave(f'gif_out/motion_iris_{index}.gif', img_list, fps=5)
    
