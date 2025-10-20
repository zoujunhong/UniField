import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import os
import random

class FlowBench_2D_LDC(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/zoujunhong/mnt/data/FlowBench/LDC_NS_2D/128x128/nurbs_lid_driven_cavity_Y.npz', num_points=1024):
        
        self.data = np.load(data_root, mmap_mode='r')
        self.len = self.data['data'].shape[0]
        self.num_points = num_points
        print('load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        """Total number of samples of data."""
        return self.len

    def __getitem__(self, idx):
        pressure = self.data['data'][idx:idx+1,2:3]
        source_grid = torch.rand([1, 1, self.num_points, 2], device=pressure.device) * 2 - 1
        points = F.grid_sample(pressure, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 12, c]
        points = torch.cat([points, source_grid.squeeze(1)], dim=-1)
        return points


class FlowBench_3D_LDC(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/zoujunhong/mnt/data/FlowBench/LDC_NS_3D/point_cloud/', num_points=8192, repeat=1, sample=False):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)[:900]
        self.len = len(self.file_list)
        self.num_points = num_points
        self.repeat = repeat
        self.sample = sample
        self.re = np.loadtxt('/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_3D_Re.txt')
        print('FlowBench3D: load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        """Total number of samples of data."""
        return self.len * self.repeat

    def __getitem__(self, idx):
        data = np.loadtxt(osp.join(self.data_root, self.file_list[idx//self.repeat]))
        data = torch.from_numpy(data).float()
        data[:,3] = data[:,3] * 10.
        data[:,2] -= torch.min(data[:,2])
        
        N = data.shape[0]
        rand_idx = torch.randperm(N)
        data = data[rand_idx, :]  # [B, N, 3]
        data = data[:self.num_points]
        cond = torch.tensor([self.re[idx//self.repeat],0], dtype=torch.float32)
        
        # cd = torch.tensor(float('nan'), dtype=torch.float32)
        cd = torch.tensor((float('nan'),float('nan'),float('nan'),float('nan'),float('nan')), dtype=torch.float32)
        
        return data[:,:3], data[:,3], cond, torch.tensor(2).long(), cd

if __name__ == '__main__':
    dataset = FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=1)
    max1 = -10000
    min1 = 10000
    for i in range(1000):
        data = dataset.__getitem__(i)
        
        if torch.max(data[:,0]) > max1:
            max1 = torch.max(data[:,0])
            
        if torch.min(data[:,0]) < min1:
            min1 = torch.min(data[:,0])
        print(max1, min1)