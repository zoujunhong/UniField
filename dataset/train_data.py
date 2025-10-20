import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import os
import random

class Traindata_3D_linear(torch.utils.data.Dataset):
    def __init__(self, repeat=1000, split='train', num_points=10000, train_samples=36):
        self.data_root1 = '/data/home/zdhs0017/zoujunhong/data/Train/point_cloud'
        self.data_root2 = '/data/home/zdhs0017/zoujunhong/data/Train/point_cloud_sample'
        self.file_list = [
            'p_75.0_0.txt', 'p_75.0_5.txt',  'p_75.0_10.txt', 'p_75.0_15.txt',
            'p_77.8_0.txt', 'p_77.8_5.txt',  'p_77.8_10.txt', 'p_77.8_15.txt', 
            'p_80.6_0.txt', 'p_80.6_5.txt',  'p_80.6_10.txt', 'p_80.6_15.txt',
            'p_83.3_0.txt', 'p_83.3_5.txt',  'p_83.3_10.txt', 'p_83.3_15.txt', 
            'p_86.1_0.txt', 'p_86.1_5.txt',  'p_86.1_10.txt', 'p_86.1_15.txt',
            'p_88.9_0.txt', 'p_88.9_5.txt',  'p_88.9_10.txt', 'p_88.9_15.txt', 
            'p_91.7_0.txt', 'p_91.7_5.txt',  'p_91.7_10.txt', 'p_91.7_15.txt', 
            'p_94.4_0.txt', 'p_94.4_5.txt',  'p_94.4_10.txt', 'p_94.4_15.txt', 
            'p_97.2_0.txt', 'p_97.2_5.txt',  'p_97.2_10.txt', 'p_97.2_15.txt']
        
        self.train_list = [
            'p_75.0_0.txt', 'p_75.0_5.txt',  'p_75.0_10.txt', 'p_75.0_15.txt',
            'p_77.8_0.txt', 'p_77.8_5.txt',  'p_77.8_10.txt', 'p_77.8_15.txt', 
            'p_80.6_0.txt', 'p_80.6_5.txt',  'p_80.6_10.txt', 'p_80.6_15.txt',
            'p_83.3_0.txt', 'p_83.3_5.txt',  'p_83.3_10.txt', 'p_83.3_15.txt', 
            'p_86.1_0.txt', 'p_86.1_5.txt',  'p_86.1_10.txt', 'p_86.1_15.txt',
            'p_88.9_0.txt', 'p_88.9_5.txt',  'p_88.9_10.txt', 'p_88.9_15.txt', 
            'p_91.7_0.txt', 'p_91.7_5.txt',  'p_91.7_10.txt', 'p_91.7_15.txt', 
            'p_94.4_0.txt', 'p_94.4_5.txt',  'p_94.4_10.txt', 'p_94.4_15.txt', 
            'p_97.2_0.txt', 'p_97.2_5.txt',  'p_97.2_10.txt', 'p_97.2_15.txt']
        
        # self.train_list = random.sample(self.file_list, train_samples)
        
        # self.train_list = [
        #     'p_75.0_0.txt', 'p_75.0_5.txt',  'p_75.0_10.txt',
        #     'p_77.8_0.txt',                  'p_77.8_10.txt', 'p_77.8_15.txt', 
        #     'p_80.6_0.txt', 'p_80.6_5.txt',  'p_80.6_10.txt',
        #     'p_83.3_0.txt', 'p_83.3_5.txt',  'p_83.3_10.txt', 'p_83.3_15.txt', 
        #     'p_86.1_0.txt', 'p_86.1_5.txt',  'p_86.1_10.txt', 'p_86.1_15.txt',
        #     'p_88.9_0.txt', 'p_88.9_5.txt',  'p_88.9_10.txt', 'p_88.9_15.txt', 
        #     'p_91.7_0.txt',                                   'p_91.7_15.txt', 
        #     'p_94.4_0.txt', 'p_94.4_5.txt', 
        #     'p_97.2_0.txt', 'p_97.2_5.txt',  'p_97.2_10.txt', 'p_97.2_15.txt']
        
        self.num_points = num_points
        self.pc = []
        self.pc_sample = []
        self.cond = []
        for file in self.file_list:
            if (split == 'train') and (file in self.train_list):
                self.pc.append(torch.from_numpy(np.loadtxt(osp.join(self.data_root1, file))[:]).float())
                self.pc_sample.append(torch.from_numpy(np.loadtxt(osp.join(self.data_root2, file))[:8]).float())
                train_speed = float(file[2:6])
                wind_speed = float(file[7:].split('.')[0])
                self.cond.append([train_speed, wind_speed])
            elif (split == 'val') and (not file in self.train_list):
                self.pc.append(torch.from_numpy(np.loadtxt(osp.join(self.data_root1, file))[:]).float())
                self.pc_sample.append(torch.from_numpy(np.loadtxt(osp.join(self.data_root2, file))[:8]).float())
                train_speed = float(file[2:6])
                wind_speed = float(file[7:].split('.')[0])
                self.cond.append([train_speed, wind_speed])
            
        self.len = len(self.pc)
        self.repeat = repeat
        print('Highway data: load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        return self.len * self.repeat
    
    def __getitem__(self, idx):
        pc = self.pc[idx//self.repeat].clone().float()
        pc[:,0] = pc[:,0] - 12.5
        pc[:,1] = pc[:,1] - 12.5
        pc[:,2] = pc[:,2] - 0.15
        # pc[:,:3] = pc[:,:3] / 12.5
        N = pc.shape[0]
        rand_idx = torch.randperm(N)
        pc = pc[rand_idx, :]  # [B, N, 3]
        pc = pc[:self.num_points]
        
        cond = torch.tensor(self.cond[idx//self.repeat], dtype=torch.float32)
        cd = torch.tensor((float('nan'),float('nan'),float('nan'),float('nan'),float('nan')), dtype=torch.float32)
        # cond = torch.tensor([self.cond[idx//self.repeat][0], self.cond[idx//self.repeat][1], 12.5], dtype=torch.float32)
        return pc[:,:3], pc[:,3], cond, torch.tensor(0).long(), cd


if __name__ == '__main__':
    dataset = Traindata_3D_linear(repeat=1)
    for i in range(10):
        dataset.__getitem__(i)