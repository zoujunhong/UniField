# System libs
import os
import time
import json
# import math
import random
import argparse

import matplotlib.pyplot as plt
# Numerical libs
import torch
import torch.nn.functional as F
# Our libs
from dataset.DrivAerExperiment import SurfacePressureDataset
from dataset.DrivAerNet import get_val_dataloaders, PRESSURE_MEAN, PRESSURE_STD, get_dataloaders
from dataset.DrivAerML import DrivAerML
from dataset.Flight import Wings3D
from dataset.Nga import NGADataset_with_forces
from dataset.FlowBench import FlowBench_3D_LDC
from dataset.Magnetic import Mag

from model.SlotAttnPointTransformer_MultiFlowCondAdapter import PointTransformer_MultiFlowCondAdapter as Model
from model.SlotAttnPointTransformer import PointTransformer_CondAdapter
from model.SlotAttnPointTransformer import PointTransformer_CondAdapter_CoeffPred
import numpy as np
import datetime as datetime
import cv2
from random import randint
from loss.loss import SimpleLpLoss
from matplotlib import cm

def r2_score(output, target):
    """Compute R-squared score."""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(ss_tot, ss_res)
    return r2

def save_segmented_pointcloud_txt(points, labels, filename="segmented_pointcloud.txt"):
    """
    将点云按标签着色后保存为 XYZRGB 格式的 txt 文件。

    参数：
        points: torch.Tensor 或 np.ndarray，形状 [N, 3]
        labels: torch.Tensor 或 np.ndarray，形状 [N]，整数标签
        filename: 保存路径
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    labels = labels.astype(int)
    unique_labels = np.unique(labels)

    # 使用 matplotlib colormap 映射颜色（例如 tab20）
    cmap = cm.get_cmap('tab20', len(unique_labels))
    color_dict = {label: np.array(cmap(i)[:3]) * 255 for i, label in enumerate(unique_labels)}

    # 构建 XYZRGB 数组
    rgb = np.stack([color_dict[l] for l in labels], axis=0).astype(np.uint8)
    xyz_rgb = np.hstack([points, rgb])

    # 保存为 txt
    np.savetxt(filename, xyz_rgb, fmt="%.6f %.6f %.6f %d %d %d")
    print(f"点云已保存到: {filename}")

    
def test(model, data_loader):
    model.eval()
    total_l2re = 0.
    total_l1re = 0.
    total_mse = 0.
    total_mae = 0.
    total_maxerror = 0.
    total_rmse = 0.
    count = 0.
    
    all_preds = []
    all_targets = []
    torch.set_printoptions(precision=4,sci_mode=False,linewidth=1000)
    myloss = SimpleLpLoss(size_average=False)
    for idx,data in enumerate(data_loader):
        if idx % 100 == 0:
            print(idx)
        with torch.no_grad():
            # while True:
                point_cloud, pressure, cond, route, cd = data
                # point_cloud = point_cloud.squeeze(1).permute(0,2,1).float().cuda()
                point_cloud = point_cloud.float().cuda()
                pressure = pressure.float().cuda()
                # pressure = (pressure.float().cuda() - PRESSURE_MEAN) / PRESSURE_STD
                cond = cond.float().cuda()
                route = route.cuda()
                cd = cd.cuda()
                len = pressure.shape[1]
                # N = point_cloud.shape[1]
                # rand_idx = torch.randperm(N).cuda()
                # pointcloud_shuffled = point_cloud[:, rand_idx, :]  # [B, N, 3]
                # pressure_shuffled = pressure[:, rand_idx]

                # num_chunks = N // 10000  # 完整 chunk 数量
                # total_points = num_chunks * 10000
                # pointcloud_trimmed = pointcloud_shuffled[:, :total_points, :]
                # pressure_sample = pressure_shuffled[:, :total_points]
                
                # pointcloud_chunks = pointcloud_trimmed.view(num_chunks, 10000, 3)

                # 输入到模型处理
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    rec = model(point_cloud, cond, route)  # 假设输出为 [B * num_chunks, ...]
                    # rec = ((rec * PRESSURE_STD) + PRESSURE_MEAN) / 450
                    # rec = (rec*450 - PRESSURE_MEAN) / PRESSURE_STD
                    rec = rec[:,-len:,0]

                L2RE = torch.sum((pressure - rec) ** 2)**0.5 / torch.sum(pressure ** 2)**0.5
                L1RE = torch.sum(torch.abs(pressure - rec)) / torch.sum(torch.abs(pressure))
                # print(pred_cd.shape, cd.shape)
                mse = F.mse_loss(rec, pressure)
                mae = F.l1_loss(rec, pressure)
                # print(pred_cd, cd)
                max_error = torch.max(torch.abs(rec - pressure))
                rmse = mse**0.5
                
                total_mse += mse
                total_mae += mae
                # total_maxerror = max(total_maxerror, mae.item())
                total_l2re += L2RE
                total_l1re += L1RE
                total_rmse += rmse
                count += 1
                
                # all_preds.append(pred_cd.squeeze(0).cpu().numpy())
                # all_targets.append(cd.cpu().numpy())
                # print('sample', 100*mse.item(), 10*mae.item(), max_error.item(), 100*L2RE.item(), 100*L1RE.item(), rmse.item())
                # print('total', 100000*total_mse.item()/count, 1000*total_mae.item()/count, 100*total_maxerror)
                print('total', 100*total_mse.item()/count, 10*total_mae.item()/count, 100*total_l2re/count, 100*total_l1re/count)

                # visualize_hierarchical_segmentation(point_cloud.squeeze(0), attn_list, 5000)
    # all_preds = np.concatenate(all_preds)
    # all_targets = np.concatenate(all_targets)
    # test_r2 = r2_score(torch.from_numpy(all_targets), torch.from_numpy(all_preds))
    # print(test_r2)
                # a = input()
    # print(total_rmse.item()/(idx+1), 100*total_loss.item()/(idx+1))

                
    

def main():
    # Network Builders
    model = Model(
        depth=[8,8,8,8,8],
        channels=[192,384,768,1536,3072],
        num_points=[1024, 256, 64, 16],
        k=16
    )
    
    # load nets into gpu
    to_load = torch.load('/data/home/zdhs0017/zoujunhong/model/savemodel/field/SAPT_Unified_pretrain_2b_p8192/model.pth',map_location=torch.device("cpu"))
    # to_load = torch.load('/data/home/zdhs0017/zoujunhong/model/savemodel/field/SAPT_unified_pretrain_2b_p8192_0928/model_epoch_40.pth',map_location=torch.device("cpu"))
    keys_list = list(to_load.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.module.', '')
            to_load[deal_key] = to_load[key]
            del to_load[key]
    model.load_state_dict(to_load,strict=False)
    model = model.cuda()
    
    # dataset = SurfacePressureDataset()
    # dataset = Mag(num_points=8192, repeat=1, train_samples=20)
    # dataset = FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=1, num_points=8192)
    # dataset = DrivAerML(repeat=1, num_points=8192)
    # dataset = Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', 8192, train_samples=1260, repeat=1, split='val')
    dataset = NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', 8192, verbose=False)
    loader_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False,
                                    num_workers=4)
    
    # loader_train = get_val_dataloaders(
    #     dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure',
    #     num_points=8192,
    #     batch_size=1,
    #     num_workers=4)
    
    test(model, loader_train)


    print('Testing Done!')


if __name__ == '__main__':
    main()
