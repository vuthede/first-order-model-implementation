import os
import sys
import numpy as np
import cv2

import torch
from tqdm import tqdm

# Dataset
from dataset.frames_dataset import FramesDataset
from torch.utils.data import DataLoader

# Models
from modules.keypoint_detector import KPDetector 
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import Discriminator
import torch.optim as optim

# Loss
device = 'cuda'

def load_model(ckpt):
    checkpoint = torch.load(ckpt, map_location=device)
     # Model here
    K = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5, temperature=0.1,
                   estimate_jacobian=True, scale_factor=0.25, single_jacobian_map=False, pad=0)
    
    dense_motion_params = {"block_expansion":64, "max_features": 1024, "num_blocks":5, "scale_factor":0.25}
    G = OcclusionAwareGenerator(num_channels=3, num_kp=10, block_expansion=64, max_features=512, num_down_blocks=2,
                 num_bottleneck_blocks=6, estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=True)
    
    D = Discriminator(num_channels=3, block_expansion=32, num_blocks=4, max_features=512,
                sn=True, use_kp=False, num_kp=10, kp_variance=0.01, estimate_jacobian= True)

    K.load_state_dict(checkpoint["K_state_dict"], strict=True)
    G.load_state_dict(checkpoint["G_state_dict"], strict=True)
    D.load_state_dict(checkpoint["D_state_dict"], strict=True)
    K = K.to(device)
    G = G.to(device)
    D = D.to(device)
    return K, G, D

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

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


def vis(x, x_prime, x_prime_hat, kp_src, kp_driving):
        """
        x: Bx3x1xHxW
        x_prime: Bx3x1xHxW
        x_prime_hat: Bx3x1xHxW
        kp_src: Bx1x10x2
        kp_driving: Bx1x10x2
        """
        _,_,h,w = x.shape
        x = x.detach().cpu().numpy()
        x_prime = x_prime.detach().cpu().numpy()
        x_prime_hat = x_prime_hat.detach().cpu().numpy()


        kp_src = kp_src.detach().cpu().numpy()
        kp_driving = kp_driving.detach().cpu().numpy()

        for i, (x1, x2, x3, ks, kd) in enumerate(zip(x, x_prime, x_prime_hat, kp_src, kp_driving)):
            x1 = (np.transpose(x1, (1,2,0))*255.0).astype(np.uint8)

            x2 = (np.transpose(x2, (1,2,0))*255.0).astype(np.uint8)

            x3 = (np.transpose(x3, (1,2,0))*255.0).astype(np.uint8)

            
            ks = (ks+1) * np.array([w,h]) / 2.0
            kd = (kd+1) * np.array([w,h]) / 2.0

            x1 = draw_landmarks(x1, ks)
            x2 = draw_landmarks(x2, kd)
            x3 = draw_landmarks(x3, kd)
            # x4 = draw_landmarks(x4, kd)
            img = np.hstack((x1, x2, x3))
            return img

def demo_dataset(K, G, D):
    root_dir = "../MonkeyNet/data/vox"
    augmentation_params = {}
    dataset = FramesDataset(root_dir, augmentation_params=augmentation_params, image_shape=(256, 256, 3), is_train=True,
                 random_seed=0, pairs_list=None, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
    
    dataloader = iter(dataloader)
    for i in range(96):
        data = next(dataloader)
    x_prime, x = data["video"], data["source"]
    x_prime = x_prime.to(device)   
    x = x.to(device)    
            
    # Keypoint detection
    kp_driving = K(x_prime)
    kp_src = K(x)
    # import pdb; pdb.set_trace()

    prediction = G(source_image=x, kp_driving=kp_driving, kp_source=kp_src)
    visualize(x, x_prime, prediction["video_prediction"],  prediction["video_deformed"],  kp_src["mean"], kp_driving["mean"], "./debug", i)

from scipy.spatial import ConvexHull
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new






if __name__ == '__main__':
    from skimage import io, img_as_float32

    # Load model
    ckpt = "checkpoints/baseline_vox_first_order_motion/51.pth.tar"
    K, G, D = load_model(ckpt = ckpt)
    K.eval()
    G.eval()
    D.eval()

    # demo_dataset(K, G, D)

    # Load src and video driving
    driving_path = "/home/ubuntu/vuthede/MonkeyNet/data/vox/test/Kathie_Lee_Gifford-lfrJYPLyYwM-19.jpg"
    # driving_path = "/home/ubuntu/vuthede/first-order-model-implementation/face_de1.png"

    driving = io.imread(driving_path) 
    driving = img_as_float32(driving)
    driving = np.moveaxis(driving, 1, 0)
    driving = driving.reshape((-1,) + (256,256,3))
    driving = np.moveaxis(driving, 1, 2)
    driving = np.expand_dims(driving,0)
    driving = np.transpose(driving, (0,4,1,2,3))
    
    ### Src
    src_path = "/home/ubuntu/vuthede/first-order-model-implementation/face_de.png"
    src = io.imread(src_path) 
    src = img_as_float32(src)
    src = np.moveaxis(src, 1, 0)
    src = src.reshape((-1,) + (256,256,3))
    src = np.moveaxis(src, 1, 2)
    src = np.expand_dims(src,0)
    src = np.transpose(src, (0,4,1,2,3))
    src = src[:,:,25:26,:,:]


    #### An image
    # src_path = "./trinh1.png"
    # src = cv2.imread(src_path)
    # src = cv2.resize(src, (256, 256)) #BxCxDxH,W
    # src = src/255.0
    # src = np.expand_dims(np.expand_dims(src, 0), 0) # 1x1xHxWx3
    # src  = np.transpose(src, (0,4,1,2,3)) # 1x3x1x256x256
    # import pdb; pdb.set_trace();
    out = cv2.VideoWriter(f'motion_transfer_de.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256*3, 256))

    s = torch.FloatTensor(src).to(device)
    dri = torch.FloatTensor(driving).to(device)


    dri0 = dri[:,:,0,:,:]
    kp_driving_initial = K(dri0)


    for i in range(driving.shape[2]):
        dri1 = dri[:,:,i,:,:]
        s1 = s[:,:,0,:,:]
        kp_dri = K(dri1)
        kp_s = K(s1)
        kp_dri_norm = normalize_kp(kp_s, kp_dri, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=True, use_relative_jacobian=True)
        
        # Keypoint detection
        # import pdb; pdb.set_trace();

        prediction = G(source_image=s.squeeze(2), kp_driving=kp_dri_norm, kp_source=kp_s)

        img_out = vis(s.squeeze(2), dri[:,:,i,:,:], prediction["prediction"], kp_s["value"], kp_dri["value"])
        print(img_out.shape)
        out.write(img_out)
    

    
