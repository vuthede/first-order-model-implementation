import os
import sys
import numpy as np
import cv2


import torch
from tqdm import tqdm

sys.path.insert(0, "..")
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
import onnxruntime
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

def load_generator(ckpt = "../checkpoints/motion_iris_thin_plate_spline_motion_more_control_points/15.pth.tar"):
    # Load model
    G = load_model(ckpt = ckpt)
    G.eval()
    return G

def synthesize_image(src, lmks106_2d, box, delta_x, delta_y, delta_x_right, delta_y_right):
    # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
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
    img_out = np.ascontiguousarray(img_out)
    return img_out

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


def load_dataset():
    root_dir = "../data/eth_motion_data"
    augmentation_params = {"flip_param" : {"horizontal_flip": False, "time_flip":False}}

    dataset1 = FramesDataset(root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                random_seed=0, pairs_list=None, augmentation_params=augmentation_params)
    dataset1.KEYPOINT_INDICES = list(range(106))
    dataset1.KEYPOINT_INDICES_WITH_EYELID = list(range(106))
    print(len(dataset1))

    return dataset1


def load_gazestate_model():
    gzst_model_onnx = "/home/ubuntu/vuthede/heatmap-based-landmarker/onnx_models/test_aptiv_normgaze.onnx"
    gazestate_session = onnxruntime.InferenceSession(gzst_model_onnx,  providers=['CUDAExecutionProvider'])

    return gazestate_session

def inference_gaze(gs_model, img):
    gazestate_face_shape = "224,224"
    input_shape = tuple(map(int, gazestate_face_shape.split(',')))
    cropface = cv2.resize(img, input_shape)
    cropface  = cv2.cvtColor(cropface, cv2.COLOR_BGR2GRAY)
    cropface = np.expand_dims(cropface, -1)
    cropface = np.ascontiguousarray(cropface, dtype=np.float32)
    ort_inputs = {
                gs_model.get_inputs()[0].name: cropface[None, :, :, :]
            }
    gazestate_result = gs_model.run(None, ort_inputs)
    gaze = gazestate_result[4][0][:2]
    return gaze

if __name__ == "__main__":
    import pandas as pd

    # Param
    USE_GT_MOTION = True
    pred_out_csv = "pred_using_gt_motion.csv"

    G = load_generator()
    gs_model = load_gazestate_model()
    dataset = load_dataset()
    mapper = GazeAnglePixelMapping(facelmksmodel="../devudata/face68model.txt")
    out = cv2.VideoWriter(f'motion_benchmark.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (256*3, 256))

    fx = 960
    fy = 960
    cx=224
    cy=224
    import imageio
    img_list = []
    data = {"pitch_gt":[], "yaw_gt":[], "pitch_pred":[], "yaw_pred":[], "pitch_syn_pred":[], "yaw_syn_pred":[], "pose_x":[], "pose_y":[], "pose_z":[], "pitch_src":[], "yaw_src":[]}
    for i in tqdm(range(4000)):
        # if i==1000:
            # break
        d = dataset[i]
        im1 = d["source"]
        im1 = (np.transpose(im1, (1,2,0))*255.0).astype(np.uint8)
        im1 = cv2.resize(im1, (448, 448))

        im2 = d["driving"]
        im2 = (np.transpose(im2, (1,2,0))*255.0).astype(np.uint8)
        im2 = cv2.resize(im2, (448, 448))
        lmks1 = d["lmks_source"]["value"]
        lmks2 = d["lmks_driving"]["value"]

        gaze1 = d["gaze_source"]
        gaze2 = d["gaze_driving"]
        
        
        lmks1 = (lmks1+1)*448 /2
        lmks2 = (lmks2+1)*448 /2
        # lmks1 = lmks106_to_lmks68(lmks1)
        # lmks2 = lmks106_to_lmks68(lmks2)

        x2=448
        y2=448
        x1=0
        y1=0
        scale_x = 256.0/(x2-x1)
        scale_y = 256.0/(y2-y1)
        left_del, right_del, euler_head = mapper.deltagaze_2_delta_xy(lmks106_to_lmks68(lmks1), fx, fy, cx, cy, gaze1, delta_gaze=gaze2-gaze1)
        
        if USE_GT_MOTION:
            left_iris1, right_iris1 = lmks1[104], lmks1[105]
            left_iris2, right_iris2 = lmks2[104], lmks2[105]
            left_del = left_iris2 - left_iris1
            right_del = right_iris2 - right_iris1

        # Scale delta x,y in full image to cropface 256 x256
        left_del[0] = left_del[0]*scale_x
        right_del[0] = right_del[0]*scale_x 
        left_del[1] = left_del[1]*scale_y
        right_del[1] = right_del[1]*scale_y 
    
        # left_del = left_del*2/(256)  # Norm into -1, 1
        # right_del = right_del*2/(256) # Norm into -1, 1
        left_del = left_del*2/(256)  # Norm into -1, 1
        right_del = right_del*2/(256) # Norm into -1, 1

        face_synthesized = synthesize_image(src=im1, lmks106_2d=lmks1, box=[x1,y1,x2,y2],delta_x=left_del[0], delta_y=left_del[1], delta_x_right=right_del[0], delta_y_right=right_del[1])
        # face_synthesized = cv2.cvtColor(face_synthesized, cv2.COLOR_RGB2BGR)

        gt_pred_gaze = inference_gaze(gs_model, im2)
        syn_pred_gaze = inference_gaze(gs_model, face_synthesized)
        gaze_gt = gaze2

        face_synthesized = draw_gaze(face_synthesized, (224,100), syn_pred_gaze,length=50)
        
        # original
        im1 = draw_gaze(im1, (224,100), gaze1,length=50)
        im1 = cv2.resize(im1, (256,256))
        
        # Target
        im2 = draw_gaze(im2, (224,100), gt_pred_gaze,length=50)
        im2 = cv2.resize(im2, (256,256))

        # synthesize_image
        face_synthesized = draw_gaze(face_synthesized, (224,100), syn_pred_gaze,length=50)
        face_synthesized = cv2.resize(face_synthesized, (256,256))

        concat_img = np.hstack([im1, im2, face_synthesized])
        # out.write(concat_img)
        img_list.append(concat_img)
        
        data["pitch_gt"].append(gaze2[0]*180.0/3.14)
        data["yaw_gt"].append(gaze2[1]*180.0/3.14)
        data["pitch_pred"].append(gt_pred_gaze[0]*180.0/3.14)
        data["yaw_pred"].append(gt_pred_gaze[1]*180.0/3.14)
        data["pitch_syn_pred"].append(syn_pred_gaze[0]*180.0/3.14)
        data["yaw_syn_pred"].append(syn_pred_gaze[1]*180.0/3.14)
        data["pose_x"].append(euler_head[0])
        data["pose_y"].append(euler_head[1])
        data["pose_z"].append(euler_head[2])
        data["pitch_src"].append(gaze1[0]*180.0/3.14)
        data["yaw_src"].append(gaze1[1]*180.0/3.14)
        
        print(gt_pred_gaze, syn_pred_gaze, gaze_gt, gaze1)
        # break

    df = pd.DataFrame(data)
    df.to_csv(pred_out_csv)
    imageio.mimsave(f'synthesize_benchmark.gif', img_list, fps=2)


  