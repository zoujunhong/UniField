# data_loader.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Data loading utilities for the DrivAerNet++ dataset.

This module provides functionality for loading and preprocessing point cloud data
with pressure field information from the DrivAerNet++ dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pyvista as pv
import logging
from utils_train import MultiEpochsDataLoader

class SurfacePressureDataset(Dataset):
    """
    Dataset class for loading and preprocessing surface pressure data from DrivAerNet++ VTK files.

    This dataset handles loading surface meshes with pressure field data,
    sampling points, and caching processed data for faster loading.
    """

    def __init__(self, root_dir: str, num_points: int, route: int):
        """
        Initializes the SurfacePressureDataset instance.

        Args:
            root_dir: Directory containing the VTK files for the car surface meshes.
            num_points: Fixed number of points to sample from each 3D model.
            preprocess: Flag to indicate if preprocessing should occur or not.
            cache_dir: Directory where the preprocessed files (NPZ) are stored.
        """
        self.root_dir = root_dir
        self.vtk_files = [f for f in os.listdir(root_dir) if f.endswith('.vtk')]
        self.num_points = num_points
        self.route=route

    def __len__(self):
        return len(self.vtk_files)

    def sample_point_cloud_with_pressure(self, mesh, n_points=5000):
        """
        Sample n_points from the surface mesh and get corresponding pressure values.

        Args:
            mesh: PyVista mesh object with pressure data stored in point_data.
            n_points: Number of points to sample.

        Returns:
            A tuple containing the sampled point cloud and corresponding pressures.
        """
        if mesh.n_points > n_points:
            indices = np.random.choice(mesh.n_points, n_points, replace=False)
        else:
            indices = np.arange(mesh.n_points)
            logging.info(f"Mesh has only {mesh.n_points} points. Using all available points.")

        sampled_points = mesh.points[indices]
        sampled_pressures = mesh.point_data['p'][indices]  # Assuming pressure data is stored under key 'p'
        sampled_pressures = sampled_pressures.flatten()  # Ensure it's a flat array

        return pv.PolyData(sampled_points), sampled_pressures

    def __getitem__(self, idx):
        vtk_file_path = os.path.join(self.root_dir, self.vtk_files[idx])
        mesh = pv.read(vtk_file_path)
        pressures = mesh.point_data['p']
        point_cloud, pressures = self.sample_point_cloud_with_pressure(mesh, self.num_points)


        point_cloud_np = np.array(point_cloud.points)
        point_cloud_tensor = torch.tensor(point_cloud_np, dtype=torch.float32)
        point_cloud_tensor[:, 0] = point_cloud_tensor[:, 0] - (torch.max(point_cloud_tensor[:, 0]) + torch.min(point_cloud_tensor[:, 0]))/2
        pressures_tensor = torch.tensor(pressures, dtype=torch.float32).clamp(-1351, 451) / 450
        
        v = 30
        coeff = torch.rand(1) * 3 + 0.2
        v = v * coeff
        point_cloud_tensor = point_cloud_tensor / coeff
        
        return point_cloud_tensor, pressures_tensor, torch.tensor([v, 0]), torch.tensor(0).long()


def create_subset(dataset, ids_file):
    """
    Create a subset of the dataset based on design IDs from a file.

    Args:
        dataset: The full dataset
        ids_file: Path to a file containing design IDs, one per line

    Returns:
        A Subset of the dataset containing only the specified designs
    """
    try:
        with open(ids_file, 'r') as file:
            subset_ids = [id_.strip() for id_ in file.readlines()]
        subset_files = [f for f in dataset.vtk_files if any(id_ in f for id_ in subset_ids)]
        subset_indices = [dataset.vtk_files.index(f) for f in subset_files]
        if not subset_indices:
            logging.error(f"No matching VTK files found for IDs in {ids_file}.")
        return Subset(dataset, subset_indices)
    except FileNotFoundError as e:
        logging.error(f"Error loading subset file {ids_file}: {e}")
        return None

def get_datasets(
    dataset_path: str, 
    num_points: int,
    route: int = 0):

    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        route=route
    )

    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    return train_dataset

def get_dataloaders(
    dataset_path: str, 
    num_points: int, 
    batch_size: int,
    world_size: int, 
    rank: int, 
    route: int = 0,
    num_workers: int = 4):

    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        route=route
    )

    train_dataset = create_subset(full_dataset, 'txt_files/train_design_ids.txt')

    # Distributed samplers for DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_dataloader = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        drop_last=True, num_workers=num_workers
    )

    return train_dataloader

def get_val_dataloaders(
    dataset_path: str, 
    num_points: int, 
    batch_size: int,
    route: int = 0,
    num_workers: int = 4):

    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        route=route
    )

    val_dataset = create_subset(full_dataset, 'txt_files/val_design_ids.txt')

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers
    )

    return val_dataloader
    
# Constants for normalization
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

        