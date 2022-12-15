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
import imageio
from skimage import io, img_as_float32
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/ubuntu/vuthede/heatmap-based-landmarker")
from lmks_api import Landmark106Detector

# Loss
device = 'cuda'

def load_model(ckpt):
    checkpoint = torch.load(ckpt, map_location=device)
     # Model here
    dense_motion_params = {"block_expansion":64, "max_features": 1024, "num_blocks":5, "scale_factor":0.25, "using_first_order_motion":False,"using_thin_plate_spline_motion":True}
    G = OcclusionAwareGenerator(num_channels=3, num_kp=8, block_expansion=64, max_features=512, num_down_blocks=2,
                 num_bottleneck_blocks=6, estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=True)
    
    G.load_state_dict(checkpoint["G_state_dict"], strict=True)
    G = G.to(device)
    return G

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img

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

def synthize_kp_driving(kp_src, delta_x=None, delta_y=None, delta_x_right=None, delta_y_right=None):
    kp_driving = {}
    kp_driving["value"] =  kp_src["value"].clone()

    kp_driving["value"][:,-2,0] = kp_driving["value"][:,-2,0] + delta_x
    kp_driving["value"][:,-2,1] = kp_driving["value"][:,-2,1] + delta_y

    kp_driving["value"][:,-1,0] = kp_driving["value"][:,-1,0] + delta_x_right
    kp_driving["value"][:,-1,1] = kp_driving["value"][:,-1,1] + delta_y_right
    kp_driving["value_witheyelid"] =  kp_driving["value"].clone()
    return kp_driving

class GazeAnglePixelMapping():
    def __init__(self, facelmksmodel='data/face68model.txt'):
        self.facelmksmodel = facelmksmodel
        self.points_3d = self.__get_full_model_points(self.facelmksmodel)
        
    def __get_full_model_points(self, filename='data/face68model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32) / 10
        model_points = np.reshape(model_points, (3, -1)).T
        p2122 = model_points[33]
        model_points = model_points - p2122

        return model_points
    
    def find_iris_position(self, eyeballcenter, eyecenter, gaze_pitch, gaze_yaw):
        M = R.from_euler('xy', [gaze_pitch, gaze_yaw], degrees=False).as_matrix()
        iris = M@(eyecenter-eyeballcenter) + eyeballcenter
        return iris
        

    def synthesize_iris(self, eyeballcenter, eyecenter, gaze_pitch, gaze_yaw, delta_pitch=-0.2, delta_yaw=0.2):
        new_pitch = gaze_pitch + delta_pitch
        new_yaw = gaze_yaw + delta_yaw
        synthesize_iris = self.find_iris_position(eyeballcenter=eyeballcenter, eyecenter=eyecenter, gaze_pitch=new_pitch, gaze_yaw=new_yaw)
        return synthesize_iris
                 
    
    def deltagaze_2_delta_xy(self, lmks2D, fx, fy, cx, cy, gaze, delta_gaze):
        """
        /brief Return delta xy of iris given delta_gaze
        @fx, fy, cx, cy: Intrinsic parameter
        @gaze eyegaze
        @delta_gaze Change ini eyegaze
        """
        # Frontalize face and get the eyegaze in the eyeball coordinate
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((4, 1))
        ret, rvec, tvec = cv2.solvePnP(self.points_3d, lmks2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        hR = cv2.Rodrigues(rvec)[0] # rotation matrix
        euler = R.from_matrix(hR).as_euler('xyz', degrees=True)
        ht = tvec[:,0]*10 # cm to mm
        
        R_gaze = R.from_euler('xy', gaze, degrees=False).as_matrix() # rotation from gaze to camera
        R_gaze_in_eyeball = hR.T @ R_gaze
        pitch_in_eyeball =  np.arcsin(-R_gaze_in_eyeball[1][2])
        yaw_in_eyeball = np.arcsin(R_gaze_in_eyeball[0][2]/np.cos(pitch_in_eyeball))

        # Caculate eyeball center of leye, reye
        points = self.points_3d.copy()
        c_leye = points[36]/2.0 + points[39]/2.0
        l_eyeball = c_leye.copy()
        l_eyeball[2] = l_eyeball[2] + 12/10 # cm
        
        c_reye = points[42]/2.0 + points[45]/2.0
        r_eyeball = c_reye.copy()
        r_eyeball[2] = r_eyeball[2] + 12/10 # cm
        
        
        # Find position of old iris and new iris for 2 eyes
        left_iris = self.find_iris_position(eyeballcenter=l_eyeball, eyecenter=c_leye, gaze_pitch=pitch_in_eyeball, gaze_yaw=yaw_in_eyeball)
        left_syn_iris = self.synthesize_iris(eyeballcenter=l_eyeball, eyecenter=c_leye, gaze_pitch=pitch_in_eyeball, gaze_yaw=yaw_in_eyeball, delta_pitch=delta_gaze[0], delta_yaw=delta_gaze[1])
        right_iris = self.find_iris_position(eyeballcenter=r_eyeball, eyecenter=c_reye, gaze_pitch=pitch_in_eyeball, gaze_yaw=yaw_in_eyeball)
        right_syn_iris = self.synthesize_iris(eyeballcenter=r_eyeball, eyecenter=c_reye, gaze_pitch=pitch_in_eyeball, gaze_yaw=yaw_in_eyeball, delta_pitch=delta_gaze[0], delta_yaw=delta_gaze[1])
        iris_points = np.array([left_iris, left_syn_iris, right_iris, right_syn_iris])

        # Project to 2d
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros((4, 1))
        points2d, _ = cv2.projectPoints(iris_points, rvec, tvec, camera_matrix, dist_coeffs)
        points2d = np.reshape(points2d, (-1, 2)) 
        
        delta_xy_left = points2d[1] - points2d[0]
        delta_xy_right = points2d[3] - points2d[2]
        
        return delta_xy_left, delta_xy_right, euler

def load_generator(ckpt = "checkpoints/motion_iris_thin_plate_spline_motion_more_control_points/15.pth.tar"):
    # Load model
    G = load_model(ckpt = ckpt)
    G.eval()
    return G


def synthize_kp_driving(kp_src, delta_x=None, delta_y=None, delta_x_right=None, delta_y_right=None):
    kp_driving = {}
    kp_driving["value"] =  kp_src["value"].clone()

    kp_driving["value"][:,-2,0] = kp_driving["value"][:,-2,0] + delta_x
    kp_driving["value"][:,-2,1] = kp_driving["value"][:,-2,1] + delta_y

    kp_driving["value"][:,-1,0] = kp_driving["value"][:,-1,0] + delta_x_right
    kp_driving["value"][:,-1,1] = kp_driving["value"][:,-1,1] + delta_y_right
    kp_driving["value_witheyelid"] =  kp_driving["value"].clone()
    return kp_driving

def synthesize_image(src, lmks106_2d, box, delta_x, delta_y, delta_x_right, delta_y_right):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (256, 256)) #BxCxDxH,W
    src = src/255.0
    x  = np.transpose(src, (2,0,1)) # 3x256x256
    
    lmks106_2d = lmks106_2d - np.array([box[0], box[1]])

    lmks_anchors = lmks106_2d[[0, 8,16,24,32, 54,104, 105]]
    lmks_anchors = lmks_anchors*2/(box[2]-box[0]) - 1.0 # NOrm into -1--> 1
    kp_src = {"value": torch.FloatTensor(lmks_anchors), "value_witheyelid": torch.FloatTensor(lmks_anchors)}


    x = torch.FloatTensor(x)
    kp_src["value"] = torch.FloatTensor(kp_src["value"])
    x = x.to(device) 
    kp_src["value"] = kp_src["value"].to(device)
    kp_src["value_witheyelid"] = kp_src["value_witheyelid"].to(device)

    kp_src["value"].unsqueeze_(0) 
    kp_src["value_witheyelid"].unsqueeze_(0) 

    x.unsqueeze_(0) 
    kp_driving = synthize_kp_driving(kp_src, delta_x, delta_y, delta_x_right, delta_y_right)
    kp_driving["value"] = kp_driving["value"].to(device)
    kp_driving["value_witheyelid"] = kp_driving["value_witheyelid"].to(device)
    # kp_driving["value"].unsqueeze_(0) 
    # kp_driving["value_witheyelid"].unsqueeze_(0) 


    prediction = G(source_image=x, kp_driving=kp_driving, kp_source=kp_src)
    prediction = prediction["prediction"].detach().cpu().numpy()
    img_out = (np.transpose(prediction[0], (1,2,0))*255.0).astype(np.uint8)

    return img_out

def lmks2box(lmks, img, expand=1.0):
    xy = np.min(lmks, axis=0).astype(np.int32) 
    zz = np.max(lmks, axis=0).astype(np.int32)

    xy[1] = max(xy[1], 0) 
    wh = zz - xy + 1

    center = (xy + wh/2).astype(np.int32)
    # EXPAND=1.1
    EXPAND=expand
    boxsize = int(np.max(wh)*EXPAND)
    xy = center - boxsize//2
    x1, y1 = xy
    x2, y2 = xy + boxsize
    height, width, _ = img.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return [x1, y1, x2, y2]

def lmks106_to_lmks98(l):
    boundary_and_nose = list(range(0, 55))
    below_nose = list(range(58, 63))
    boundary_left_eye = list(range(66, 74))
    boundary_right_eye = list(range(75, 83))        
    mouth = list(range(84, 104))
    center_left_eye = [104]
    center_right_eye = [105]

    indice = boundary_and_nose + below_nose + boundary_left_eye + boundary_right_eye + mouth + center_left_eye +  center_right_eye
    l = np.array(l)[indice]

    return l

def lmks98_to_lmks68(l):
    LMK98_2_68_MAP = {0: 0,
                    2: 1,
                    4: 2,
                    6: 3,
                    8: 4,
                    10: 5,
                    12: 6,
                    14: 7,
                    16: 8,
                    18: 9,
                    20: 10,
                    22: 11,
                    24: 12,
                    26: 13,
                    28: 14,
                    30: 15,
                    32: 16,
                    33: 17,
                    34: 18,
                    35: 19,
                    36: 20,
                    37: 21,
                    42: 22,
                    43: 23,
                    44: 24,
                    45: 25,
                    46: 26,
                    51: 27,
                    52: 28,
                    53: 29,
                    54: 30,
                    55: 31,
                    56: 32,
                    57: 33,
                    58: 34,
                    59: 35,
                    60: 36,
                    61: 37,
                    63: 38,
                    64: 39,
                    65: 40,
                    67: 41,
                    68: 42,
                    69: 43,
                    71: 44,
                    72: 45,
                    73: 46,
                    75: 47,
                    76: 48,
                    77: 49,
                    78: 50,
                    79: 51,
                    80: 52,
                    81: 53,
                    82: 54,
                    83: 55,
                    84: 56,
                    85: 57,
                    86: 58,
                    87: 59,
                    88: 60,
                    89: 61,
                    90: 62,
                    91: 63,
                    92: 64,
                    93: 65,
                    94: 66,
                    95: 67}

    indice68lmk = np.array(list(LMK98_2_68_MAP.keys()))
    l  = np.array(l)[indice68lmk]

    return l

def lmks106_to_lmks68(l):
    l = lmks106_to_lmks98(l)
    l = lmks98_to_lmks68(l)
    return l

def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])* np.cos(pitchyaw[0])
    dy = length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

if __name__ == "__main__":
    G = load_generator()
    ckpt_lmks = "/home/ubuntu/vuthede/heatmap-based-landmarker/ckpt/epoch_80.pth.tar"
    fd_model_onnx = "/home/ubuntu/vuthede/heatmap-based-landmarker/onnx_models/fd_dms.onnx"
    gzst_model_onnx = "/home/ubuntu/vuthede/heatmap-based-landmarker/onnx_models/test_aptiv_normgaze.onnx"

    LmksDetector = Landmark106Detector(fd_model_onnx=fd_model_onnx, gzst_model_onnx=gzst_model_onnx,ckpt_lmks=ckpt_lmks) 
    # cap = cv2.VideoCapture("/home/ubuntu/vuthede/first-order-model-implementation/degaze3.webm")
    cap = cv2.VideoCapture("/home/ubuntu/vuthede/first-order-model-implementation/deread.webm")
    
    
    mapper = GazeAnglePixelMapping(facelmksmodel="./devudata/face68model.txt")
    out = cv2.VideoWriter(f'motion_fullflow_video_relative1.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640*2, 480))

    i = 0
    gaze_first_frame = np.array([0.0,0.0])
    framr_anchor_gaze=False
    while True:
        i +=1
        print(i)
        # if i>=200:
        #     break
        ret, frame = cap.read()
        

        if not ret: 
            break
        fx = frame.shape[1]
        fy = frame.shape[1]
        cx = fx//2
        cy = frame.shape[0]//2

        lmks_106, gaze = LmksDetector.detect_lmks_gaze(frame)
        print(lmks_106, gaze)
        if (i==120 or framr_anchor_gaze==False) and gaze is not None :
            gaze_first_frame = gaze
            framr_anchor_gaze = True


        if lmks_106 is not None:
            x1,y1,x2,y2 = lmks2box(lmks_106, frame,expand=1.2)
            face = frame[y1:y2, x1:x2]
            scale_x = 256.0/(x2-x1)
            scale_y = 256.0/(y2-y1)

            # Look at the camera
            relative_gaze = gaze - gaze_first_frame
            relative_gaze = np.array([0,0])
            left_del, right_del, euler_head = mapper.deltagaze_2_delta_xy(lmks106_to_lmks68(lmks_106), fx, fy, cx, cy, gaze, delta_gaze=(-gaze+np.array([0.0,0.0])) + relative_gaze)
            
            # Scale delta x,y in full image to cropface 256 x256
            left_del[0] = left_del[0]*scale_x
            right_del[0] = right_del[0]*scale_x 
            left_del[1] = left_del[1]*scale_y
            right_del[1] = right_del[1]*scale_y 
        
            left_del = left_del*2/(256)  # Norm into -1, 1
            right_del = right_del*2/(256) # Norm into -1, 1

            face_synthesized = synthesize_image(src=face, lmks106_2d=lmks_106, box=[x1,y1,x2,y2],delta_x=left_del[0], delta_y=left_del[1], delta_x_right=right_del[0], delta_y_right=right_del[1])
            face_synthesized = cv2.cvtColor(face_synthesized, cv2.COLOR_RGB2BGR)

            face = cv2.resize(face, (128,128))
            face_synthesized_128 = cv2.resize(face_synthesized, (128,128))

            frame_syn = frame.copy()
            face_synthesized_fit = cv2.resize(face_synthesized, (x2-x1,y2-y1))
            frame_syn[y1:y2, x1:x2] = face_synthesized_fit 

            concat_face = np.hstack([face, face_synthesized_128])
            frame[0:128,0:128*2] = concat_face

            concat_frame = np.hstack([frame, frame_syn])
            # print(concat_frame.shape)

            # cv2.imwrite("concat_syn_fullflow.png", concat_face)
            out.write(concat_frame)
    out.release()
            
