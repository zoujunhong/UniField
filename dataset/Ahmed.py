import torch
import os.path as osp
import numpy as np
import os

class Ahmed(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root='/data/home/zdhs0017/zoujunhong/data/Ahmed/point_cloud',
        num_points=1024,
        repeat=100,
        sample=True,
        force_root='/data/home/zdhs0017/zoujunhong/data/Ahmed',
        index_offset=0,        # 如果 run_/force_mom_ 从1开始，设为 1
        return_lift=False,     # True 时也返回 cl
    ):
        self.data_root = data_root
        self.file_list = sorted(os.listdir(data_root))
        self.len = len(self.file_list)
        self.num_points = num_points
        self.repeat = repeat
        self.sample = sample

        self.force_root = force_root
        self.index_offset = index_offset
        self.return_lift = return_lift

        print('Ahmed: load {} fields snapshot for training'.format(self.__len__()))

    def __len__(self):
        return self.len * self.repeat

    def sampling(self, point_cloud: torch.Tensor):
        N = point_cloud.shape[0]
        if N > self.num_points:
            rand_idx = torch.randperm(N)
            point_cloud = point_cloud[rand_idx, :]
            return point_cloud[:self.num_points]
        elif N < self.num_points:
            idx = torch.randint(0, N, (self.num_points,), device=point_cloud.device)
            return point_cloud[idx]

    def _read_cd_cl(self, i: int):
        """读取 cd, cl。缺失或解析失败时返回 NaN。"""
        run_idx = i + self.index_offset
        fpath = osp.join(self.force_root, f'run_{run_idx}', f'force_mom_{run_idx}.csv')

        if not osp.exists(fpath):
            return float('nan'), float('nan')

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) < 2:
                return float('nan'), float('nan')
            vals = [v for v in lines[1].replace(' ', '').split(',') if v != '']
            cd = float(vals[0]) if len(vals) >= 1 else float('nan')
            cl = float(vals[1]) if len(vals) >= 2 else float('nan')
            return cd, cl
        except Exception:
            # 任意解析异常都当作缺失
            return float('nan'), float('nan')

    def __getitem__(self, idx):
        base_idx = idx % self.len

        # 读取点云
        path = osp.join(self.data_root, self.file_list[base_idx], 'boundary.txt')
        data = np.loadtxt(path)
        data = torch.from_numpy(data).float()[:, [0, 1, 2, 4]]

        # 与原逻辑一致的几何处理
        data[:, 0] -= torch.mean(data[:, 0])
        data[:, 1] -= torch.mean(data[:, 1])
        data[:, 2] -= torch.min(data[:, 2])

        # 采样/补点
        data = self.sampling(data)

        # 读取 cd / cl（允许 NaN）
        cd, cl = self._read_cd_cl(base_idx)
        cd = torch.tensor(cd, dtype=torch.float32)
        cl = torch.tensor(cl, dtype=torch.float32)

        # print(torch.max(data[:, 0]), torch.min(data[:, 0]))
        # print(torch.max(data[:, 1]), torch.min(data[:, 1]))
        # print(torch.max(data[:, 2]), torch.min(data[:, 2]))
        # print(torch.max(data[:, 3]), torch.min(data[:, 3]))
        # print(cd)
        # 现在默认只返回 cd；需要 cl 就 return_lift=True
        if self.return_lift:
            return data[:, :3], data[:, 3], torch.tensor([38.89, 0]), torch.tensor(0).long(), cd, cl
        else:
            return data[:, :3], data[:, 3], torch.tensor([38.89, 0]), torch.tensor(0).long(), cd
