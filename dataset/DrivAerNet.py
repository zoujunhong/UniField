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
import torch.distributed as dist
import pyvista as pv
import logging
from utils_train import MultiEpochsDataLoader
import pandas as pd

class SurfacePressureDataset(Dataset):
    """
    Dataset class for loading and preprocessing surface pressure data from DrivAerNet++ VTK files.

    This dataset handles loading surface meshes with pressure field data,
    sampling points, and caching processed data for faster loading.
    """

    def __init__(self, root_dir: str, num_points: int, csv_file: str, preprocess=False, cache_dir=None):
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
        self.preprocess = preprocess
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "processed_data")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

    def __len__(self):
        return len(self.vtk_files)

    def _get_cache_path(self, vtk_file_path):
        """Get the corresponding .npz file path for a given .vtk file."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
        return os.path.join(self.cache_dir, base_name)

    def _save_to_cache(self, cache_path, point_cloud, pressures):
        """Save preprocessed point cloud and pressure data into an npz file."""
        np.savez_compressed(cache_path, points=point_cloud.points, pressures=pressures)

    def _load_from_cache(self, cache_path):
        """Load preprocessed point cloud and pressure data from an npz file."""
        data = np.load(cache_path)
        point_cloud = pv.PolyData(data['points'])
        pressures = data['pressures']
        return point_cloud, pressures

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
        vtk_file_path = os.path.join(self.root_dir, self.vtk_files[idx]) # self.vtk_files[idx]
        file_name = self.vtk_files[idx].split('.')[0]
        row = self.data_frame.loc[self.data_frame["ID"] == file_name]
        Cd = row["Drag_Value"].values[0]
        # cache_path = self._get_cache_path(vtk_file_path)
        mesh = pv.read(vtk_file_path)
        pressures = mesh.point_data['p']
        point_cloud, pressures = self.sample_point_cloud_with_pressure(mesh, self.num_points)
        # Check if the data is already cached
        # if os.path.exists(cache_path):
        #     logging.info(f"Loading cached data from {cache_path}")
        #     point_cloud, pressures = self._load_from_cache(cache_path)
        # else:
        #     if self.preprocess:
        #         logging.info(f"Preprocessing and caching data for {vtk_file_path}")
        #         try:
        #             mesh = pv.read(vtk_file_path)
        #         except Exception as e:
        #             logging.error(f"Failed to load VTK file: {vtk_file_path}. Error: {e}")
        #             return None, None  # Skip the file and return None

        #         point_cloud, pressures = self.sample_point_cloud_with_pressure(mesh, self.num_points)

        #         # Cache the sampled data to a new file
        #         self._save_to_cache(cache_path, point_cloud, pressures)
        #     else:
        #         logging.error(f"Cache file not found for {vtk_file_path} and preprocessing is disabled.")
        #         return None, None  # Return None if preprocessing is disabled and cache doesn't exist

        point_cloud_np = np.array(point_cloud.points)
        point_cloud_tensor = torch.tensor(point_cloud_np, dtype=torch.float32)
        point_cloud_tensor[:, 0] = point_cloud_tensor[:, 0] - (torch.max(point_cloud_tensor[:, 0]) + torch.min(point_cloud_tensor[:, 0]))/2
        pressures_tensor = torch.tensor(pressures, dtype=torch.float32).clamp(-1351, 451) / 450
        
        v = 30
        coeff = torch.rand(1) * 3 + 0.2
        v = v * coeff
        point_cloud_tensor = point_cloud_tensor / coeff
        cd = torch.tensor((Cd,float('nan'),float('nan'),float('nan'),float('nan')), dtype=torch.float32)
        
        return point_cloud_tensor, pressures_tensor, torch.tensor([v, 0]), torch.tensor(0).long(), cd


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
    cache_dir: str = None) -> tuple:

    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        csv_file="/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/DrivAerNetPlusPlus_Cd_8k_Updated.csv",
        preprocess=True,
        cache_dir=cache_dir
    )

    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    return train_dataset

def get_dataloaders(
    dataset_path: str, 
    num_points: int, 
    batch_size: int,
    world_size: int, 
    rank: int, 
    cache_dir: str = None, 
    num_workers: int = 4) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path: Path to the directory containing VTK files
        subset_dir: Directory containing train/val/test split files
        num_points: Number of points to sample from each mesh
        batch_size: Batch size for dataloaders
        world_size: Total number of processes for distributed training
        rank: Current process rank
        cache_dir: Directory to store processed data
        num_workers: Number of workers for data loading

    Returns:
        A tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        csv_file="/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/DrivAerNetPlusPlus_Cd_8k_Updated.csv",
        preprocess=True,
        cache_dir=cache_dir
    )

    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
    # val_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'val_design_ids.txt'))
    # test_dataset = create_subset(full_dataset, os.path.join(subset_dir, 'test_design_ids.txt'))

    # Distributed samplers for DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, num_replicas=world_size, rank=rank
    # )
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset, num_replicas=world_size, rank=rank
    # )

    train_dataloader = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        drop_last=True, num_workers=num_workers
    )
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=batch_size, sampler=val_sampler,
    #     drop_last=True, num_workers=num_workers
    # )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=batch_size, sampler=test_sampler,
    #     drop_last=True, num_workers=num_workers
    # )

    return train_dataloader#, val_dataloader, test_dataloader

def get_val_dataloaders(
    dataset_path: str, 
    num_points: int, 
    batch_size: int,
    cache_dir: str = None, 
    num_workers: int = 4) -> tuple:

    full_dataset = SurfacePressureDataset(
        root_dir=dataset_path,
        num_points=num_points,
        csv_file="/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/DrivAerNetPlusPlus_Cd_8k_Updated.csv",
        preprocess=True,
        cache_dir=cache_dir
    )

    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers
    )


    return val_dataloader# , test_dataloader
    
# Constants for normalization
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

if __name__ == '__main__':
    full_dataset = SurfacePressureDataset(
        root_dir='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure',
        num_points=32768,
        csv_file="/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/DrivAerNetPlusPlus_Cd_8k_Updated.csv"
    )

    # val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    
    for i in range(full_dataset.__len__()):
        point_cloud, pressure, _, _, _ = full_dataset.__getitem__(i)
        