import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from dataset.augmentation import AllAugmentationTransform
import glob
import cv2

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    # KEYPOINT_INDICES = [104, 105]  # Iris center
    KEYPOINT_INDICES = [0, 8, 16, 24, 32, 54, 104, 105]  # Boundary+nose points for control shape +  Iris center


    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __load_lmks(self, path, img_size=256.0):
        f_name_base = os.path.basename(path)
        if self.is_train:
            anno_f = f'{os.path.dirname(path).replace("train", "train_lmks_annotations")}'
            anno_f = f'{anno_f}/{f_name_base}'.replace(".png", ".txt")
            with open(anno_f, 'r') as f:
                lines = f.readlines()
                lmks_all = []
                for line in lines:
                    line = line.strip().rstrip()
                    lmks = list(map(float, line.split(",")))
                    lmks = np.array(lmks, dtype='float32')
                    lmks = np.reshape(lmks, (-1, 2))[self.KEYPOINT_INDICES]
                    lmks = (lmks*2.0)/img_size - 1.0 # -1 -> 1
                    lmks_all.append(lmks)
            lmks_all = np.array(lmks_all, dtype='float32')
            return lmks_all
        return None

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
        # import pdb;pdb.set_trace()
        if self.is_train and os.path.isdir(path):
            # print(path)
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            lmks = self.__load_lmks(path)
            lmks = lmks[frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            # if (video_array.shape != (20, 256, 256, 3)):
            #     print(path)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]
            lmks = self.__load_lmks(path)
            lmks = lmks[frame_idx]
        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['lmks_source'] = {"value": lmks[0]}
            out['lmks_driving'] = {"value": lmks[1]}

        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


def draw_landmarks(img, lmks, color =(0,255,0)):
    default_color = color
    for a in lmks:
        color = default_color
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 2, color, -1, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    # root_dir = "../../MonkeyNet/data/vox"
    root_dir = "../data/eth_motion_data"
    augmentation_params = {"flip_param" : {"horizontal_flip": False, "time_flip":False}, "jitter_param" :{"brightness":0.1, "contrast":0.1, "saturation":0.1, "hue":0.1}}

    dataset1 = FramesDataset(root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=augmentation_params)
    from tqdm import tqdm
    print(f'Len1 dataset :{len(dataset1)}')
    for d in dataset1:
        # print("hihi")
        # print(d["driving"].shape, d["source"].shape, d["lmks_driving"].shape,d["lmks_source"].shape)

        driving = (d["driving"].transpose(1,2,0)*255.0).astype(np.uint8)
        src = (d["source"].transpose(1,2,0)*255.0).astype(np.uint8)
        driving = np.ascontiguousarray(driving)
        src = np.ascontiguousarray(src)
        driving = draw_landmarks(driving, (d["lmks_driving"]["value"]+1)*255/2 )
        src = draw_landmarks(src, (d["lmks_source"]["value"]+1)*255/2)
        src = draw_landmarks(src, (d["lmks_driving"]["value"]+1)*255/2, color=(0,0,255))
        im = np.hstack((src, driving))
        cv2.imwrite('both.png', im)

        break
   
   
    # data = dataset[100]
    # video = data["driving"] # 3xHxW
    # src = data["source"] # 3xHxW
    # name = data["name"]
    # import pdb; pdb.set_trace();
    # print(f'Driving :{video.shape}, src shape :{src.shape}, name: {name}')

