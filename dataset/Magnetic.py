import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import os
import random

class Mag(torch.utils.data.Dataset):
    def __init__(self, repeat=1000, num_points=10000, train_samples=20):
        self.data_root = '/data/home/zdhs0017/zoujunhong/data/Magnetic/pc_head'
        self.file_list = os.listdir(self.data_root)
        self.file_list  = [file for file in self.file_list if "shape1" in file]
        # self.file_list = random.sample(self.file_list, train_samples)
        
        self.num_points = num_points
        self.pc = []
        self.pc_sample = []
        self.cond = []
        for file in self.file_list:
                train_speed = float(file.split('_')[1])
                wind_speed = float(file.split('_')[2][:-4])
                # if wind_speed > 0:
                #     continue
                self.pc.append(torch.from_numpy(np.loadtxt(osp.join(self.data_root, file))[:]).float())
                self.cond.append([train_speed, wind_speed])
            
        self.len = len(self.pc)
        self.repeat = repeat
        print('Maglev data: load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        return self.len * self.repeat
    
    def __getitem__(self, idx):
        pc = self.pc[idx//self.repeat].clone().float()
        pc[:, 0] = pc[:, 0] - (torch.max(pc[:, 0]) + torch.min(pc[:, 0]))/2
        pc[:, 1] = pc[:, 1] - (torch.max(pc[:, 1]) + torch.min(pc[:, 1]))/2
        pc[:, 2] = pc[:, 2] -  torch.min(pc[:, 2])
        pc[:,:3] = pc[:,:3] * 10
        
        # scale = torch.max(pc[:,:3])
        # pc[:,:3] = pc[:,:3] / scale
        
        N = pc.shape[0]
        rand_idx = torch.randperm(N)
        pc = pc[rand_idx, :]  # [B, N, 3]
        pc = pc[:self.num_points]
        cond = torch.tensor(self.cond[idx//self.repeat], dtype=torch.float32)
        
        cd = torch.tensor((float('nan'),float('nan'),float('nan'),float('nan'),float('nan')), dtype=torch.float32)
        # cond = torch.tensor((self.cond[idx//self.repeat][0], self.cond[idx//self.repeat][1], scale), dtype=torch.float32)
        return pc[:,:3], pc[:,3], cond, torch.tensor(0).long(), cd
