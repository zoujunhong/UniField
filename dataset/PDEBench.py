import torch
import torch.nn.functional as F
import os.path as osp
import h5py 

class PDEBench_pointcloud(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/zoujunhong/mnt/data/PDEBench/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5', split='train', num_points=1024):
        self.field_root = data_root
        f = h5py.File(self.field_root)
        self.len = int(f['Vx'].shape[0] * f['Vx'].shape[1])
        self.num_points = num_points
        self.data_idxs = []
        for i in range(self.len):
            if i % 10 > 0 and split == 'train':
                self.data_idxs.append(i)
            elif i % 10 == 0 and split == 'val':
                self.data_idxs.append(i)
        print('load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_idxs)

    def __getitem__(self, i):
        idx = self.data_idxs[i]
        sample_idx = idx // 21
        timestep_idx = idx % 21
        f = h5py.File(self.field_root)
        pressure = torch.from_numpy(f['pressure'][sample_idx:sample_idx+1, timestep_idx:timestep_idx+1])
        source_grid = torch.rand([1, 1, self.num_points, 2], device=pressure.device) * 2 - 1
        points = F.grid_sample(pressure, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        points = points - torch.mean(points)
        points = points / torch.max(torch.abs(points))
        points = torch.cat([source_grid.squeeze(1), torch.zeros([1, self.num_points, 1]), points], dim=-1)
        f.close()
        return points.squeeze(0)

class PDEBench_grid(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/zoujunhong/mnt/data/PDEBench/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5', split='train'):
        self.field_root = data_root
        f = h5py.File(self.field_root)
        len = int(f['Vx'].shape[0] * f['Vx'].shape[1])
        self.data_idxs = []
        for i in range(len):
            if i % 10 > 0 and split == 'train':
                self.data_idxs.append(i)
            elif i % 10 == 0 and split == 'val':
                self.data_idxs.append(i)
                
        print('load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_idxs)

    def __getitem__(self, i):
        idx = self.data_idxs[i]
        sample_idx = idx // 21
        timestep_idx = idx % 21
        f = h5py.File(self.field_root)
        pressure = torch.from_numpy(f['pressure'][sample_idx, timestep_idx:timestep_idx+1])
        f.close()
        return pressure


class PDEBench(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/zoujunhong/mnt/data/PDEBench/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5', split='train'):
        self.field_root = data_root
        f = h5py.File(self.field_root)
        self.len = int(f['Vx'].shape[0] * f['Vx'].shape[1])
                
        print('load {} fields snapshot for training'.format(self.__len__()))
        
    def __len__(self):
        """Total number of samples of data."""
        return self.len

    def __getitem__(self, i):
        idx = i
        f = h5py.File(self.field_root)
        pressure = torch.from_numpy(f['pressure'][idx])
        f.close()
        return pressure


if __name__ == '__main__':
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    num_frames = 21
    dataset = PDEBench()
    for i in range(1):
        x = dataset.__getitem__(random.randint(0,9999))
    
        # 假设你的时序场数据，21 帧，每一帧是 128x128
    
        field_data = x.cpu().numpy()  # 用随机数据代替实际场数据

        # 创建图像对象
        fig, ax = plt.subplots(figsize=(5, 5))
        img = ax.imshow(field_data[0], cmap='viridis', origin='lower')
        ax.axis('off')  # 关闭坐标轴

        # 更新函数
        def update(frame):
            img.set_array(field_data[frame])
            return [img]

        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)

        # 保存为 GIF 或 MP4
        # ani.save("field_animation.gif", writer="pillow", fps=10)  # 生成 GIF
        ani.save("field_animation.mp4", writer="ffmpeg", fps=10)  # 生成 MP4

        plt.show()

