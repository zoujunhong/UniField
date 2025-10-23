import torch
import os.path as osp
import numpy as np
import os


class FlowBench_3D_LDC(torch.utils.data.Dataset):
    def __init__(self, data_root='/path/to/FlowBench/LDC_NS_3D/point_cloud/', num_points=8192, repeat=1, split='train', route=2):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)[:900] if split == 'train' else os.listdir(data_root)[900:]
        self.len = len(self.file_list)
        self.num_points = num_points
        self.repeat = repeat
        self.route = route
        self.re = np.loadtxt('txt_files/LDC_3D_Re.txt')
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
        
        return data[:,:3], data[:,3], cond, torch.tensor(self.route).long()
