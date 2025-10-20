# data_loader.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Data loading utilities for the DrivAerNet++ dataset.

This module provides functionality for loading and preprocessing point cloud data
with pressure field information from the DrivAerNet++ dataset.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import pyvista as pv

class SurfacePressureDataset(Dataset):
    """
    Dataset class for loading and preprocessing surface pressure data from DrivAerNet++ VTK files.

    This dataset handles loading surface meshes with pressure field data,
    sampling points, and caching processed data for faster loading.
    """

    def __init__(self,):
        self.stl_files = [
            '/data/home/zdhs0017/NotchGeometry.stl',
            '/data/home/zdhs0017/FastGeometry.stl',
            '/data/home/zdhs0017/EstateGeometry.stl',
            ]
        self.txt_files = [
            '/data/home/zdhs0017/NOTCHBACK.txt',
            '/data/home/zdhs0017/FASTBACK.txt',
            '/data/home/zdhs0017/ESTATEBACK.txt',
            ]
        self.num_points = 8192

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        stl_file_path = self.stl_files[idx] # self.vtk_files[idx]
        mesh = pv.read(stl_file_path)
        
        txt_file_path = self.txt_files[idx]
        measurement = np.loadtxt(txt_file_path)
        mpoint = measurement.shape[0]
        
        indices = np.random.choice(mesh.n_points, self.num_points - mpoint, replace=False)
        pc = mesh.points[indices] / 1000
        pc[:, 0] = pc[:, 0] - (np.max(pc[:, 0]) + np.min(pc[:, 0]))/2
        pc[:, 2] = pc[:, 2] - np.min(pc[:, 2])
        
        pc = np.concatenate((pc, measurement[:,:3]*4))
        meas = measurement[:,3] * 450
        return pc, meas, torch.tensor([10, 0]), torch.tensor(0).long(), 0

# dataset = SurfacePressureDataset()
# pc, meas = dataset.__getitem__(0)
# print(pc.shape, meas.shape)