# data_loader.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Data loading utilities for the DrivAerNet++ dataset.

This module provides functionality for loading and preprocessing point cloud data
with pressure field information from the DrivAerNet++ dataset.
"""
import re
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torch.distributed as dist
import pyvista as pv
import logging
import random

class Wings3D(Dataset):

    def __init__(self, root_dir: str, num_points: int, split: str = 'train', repeat=100, train_samples=140):
        """
        Initializes the SurfacePressureDataset instance.

        Args:
            root_dir: Directory containing the VTK files for the car surface meshes.
            num_points: Fixed number of points to sample from each 3D model.
            preprocess: Flag to indicate if preprocessing should occur or not.
            cache_dir: Directory where the preprocessed files (NPZ) are stored.
        """
        self.root_dir = root_dir
        self.vtk_files = []
        count = 0
        self.repeat = repeat
        f_list = sorted(os.listdir(root_dir))
        for f in f_list:
            if f.endswith('.vtk'):
        #         if (count % 5 == 4 and split == 'val') or (count % 5 != 4 and split == 'train'):
                    self.vtk_files.append(os.path.join(root_dir, f))
        #         count += 1
        
        self.vtk_files = random.sample(self.vtk_files[:1260], train_samples) if split == 'train' else self.vtk_files[1260:]
        self.num_points = num_points
        print('Wings: load {} meshes for {}.'.format(len(self.vtk_files), split))
        
        
    def __len__(self):
        return len(self.vtk_files) * self.repeat
    
    def extract_m_a(self, filename):
        """
        从类似 'Jiyi_12_0_m06a-15.vtk' 形式的文件名中提取 m 和 a。
        返回: (m, a)，均为 float 类型
        """
        match = re.search(r'm(\d+)(?:a|-?a)(-?\d+)', filename)
        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        
        m_str, a_str = match.groups()
        m = float(m_str) / 10  # 因为 m 后面是 06 -> 0.6
        a = float(a_str)
        return m, a
    
    def sample(self, point_cloud):
        N = point_cloud.shape[0]
        if N > self.num_points:
            rand_idx = torch.randperm(N)
            point_cloud = point_cloud[rand_idx, :]
            return point_cloud[:self.num_points]
        elif N < self.num_points:
            idx = torch.randint(0, N, (self.num_points,), device=point_cloud.device)
            return point_cloud[idx]


    def __getitem__(self, idx):
        vtk_file_path = self.vtk_files[idx//self.repeat]
        mesh = pv.read(vtk_file_path)
        points = mesh.points #(N，3)
        # faces = mesh.faces.reshape(-1,4)[:,1:4] #(M,3)
        cp = mesh.point_data["Cp"]
        m, a = self.extract_m_a(vtk_file_path)
        
        point_cloud_tensor = torch.tensor(points, dtype=torch.float32)
        point_cloud_tensor[:, 0] = point_cloud_tensor[:, 0] - (torch.max(point_cloud_tensor[:, 0]) + torch.min(point_cloud_tensor[:, 0]))/2
        point_cloud_tensor[:, 1] = point_cloud_tensor[:, 1] - (torch.max(point_cloud_tensor[:, 1]) + torch.min(point_cloud_tensor[:, 1]))/2
        point_cloud_tensor[:, 2] = point_cloud_tensor[:, 2] -  torch.min(point_cloud_tensor[:, 2])

        pressures_tensor = torch.tensor(cp, dtype=torch.float32)
        point_cloud = self.sample(torch.cat([point_cloud_tensor, pressures_tensor[:,None]], dim=1))
        
        cd = torch.tensor((float('nan'),float('nan'),float('nan'),float('nan'),float('nan')), dtype=torch.float32)
        
        return point_cloud[:,:3], point_cloud[:,3], torch.tensor([m, a]), torch.tensor(1).long(), cd

if __name__ == '__main__':
    dataset = Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', 7200)
    # x_min = 1000000
    # for i in range(1400):
    #     shape = dataset.__getitem__(i)
    #     if shape < x_min:
    #         x_min = shape
    
    # print(x_min)