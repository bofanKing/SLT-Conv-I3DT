# 文件: signjoey/i3d_video_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import glob

class I3DVideoDataset(Dataset):
    def __init__(self, frame_root, file_list, transform=None, num_frames=None): # 初始化， 读取文件路径-一个txt文件包含name，将文件内容裁剪成224*224的格式 或者处理为TensorFlow
        self.frame_root = frame_root
        self.sample_names = self._load_file_list(file_list)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)), # 将其变成[3,H,W]的张量
            T.ToTensor(),
        ])
        self.num_frames = num_frames

    def _load_file_list(self, file_path): # 读取文件名称
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sample_names = [line.strip().split('|')[0] for line in lines if line.strip()]
        return sample_names

    def __len__(self): # 获取一个文件内的总帧数
        return len(self.sample_names)

    def __getitem__(self, idx): # 加载一个视频的全部内容 很多很多帧
        sample_name = self.sample_names[idx]
        frame_dir = os.path.join(self.frame_root, sample_name)
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frame_dir}")
        selected_frames = [self.transform(Image.open(p)) for p in frame_paths]
        clip = torch.stack(selected_frames, dim=0)  # (T, 3, H, W) 在时间维度上进行堆叠 为了符合i3d的要求
        return clip, sample_name




