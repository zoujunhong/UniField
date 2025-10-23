import numpy as np
import os
np.set_printoptions(precision=2)
import time
    
t1 = time.time()
obj_region = np.load('LDC_NS_3D/LDC_3d_X.npz')['data']
obj_region = obj_region[:, 0:1]
obj_region[obj_region > 0] = 255
obj_region[obj_region <= 0] = 0

pressure = np.load('LDC_NS_3D/LDC_3d_Y.npz')['data'][:, 2:3]
obj_region[pressure==0] = 0
data = np.concatenate((obj_region, pressure), axis=1)

def get_edge_points_numpy(tensor):
    # 假设 tensor 的形状为 (2, 128, 128, 128)
    object_region = tensor[0]  # 物体区域 (0表示物体区域, 255表示流体区域)
    flow_field = tensor[1]  # 流场值
    
    # 获取流体区域和物体区域
    fluid_region = object_region > 128  # 255表示流体区域
    object_region = object_region < 128  # 0表示物体区域
    
    # 创建一个掩码，找出流体区域与物体区域相邻的边缘点
    edge_mask = np.zeros_like(object_region, dtype=bool)
    
    # 邻接关系：考虑3x3x3邻域，查找流体区域与物体区域相邻的位置
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # 忽略中心点
                
                # 移动位置并检查邻接关系
                shifted_object_region = np.roll(object_region, shift=(dx, dy, dz), axis=(0, 1, 2))
                edge_mask = edge_mask | (fluid_region & shifted_object_region)
    
    # 获取边缘点的流场值和坐标
    edge_points = np.array(np.nonzero(edge_mask)).T  # 获取边缘点的索引 (N, 3)
    flow_values = flow_field[edge_mask]  # 获取边缘点的流场值
    
    # 归一化坐标到 [-1, 1] 范围
    normalized_coords = 2 * edge_points.astype(np.float32) / 127.0 - 1  # 将 [0, 127] 映射到 [-1, 1]
    
    # 将流场值与归一化坐标合并
    points_cloud = np.hstack((normalized_coords, flow_values[:, np.newaxis]))
    
    return points_cloud

if not os.path.exists('LDC_NS_3D/point_cloud/'):
    os.mkdir('LDC_NS_3D/point_cloud/')
    
for i in range(1000):
    pc = get_edge_points_numpy(data[i])
    
    np.savetxt('LDC_NS_3D/point_cloud/{}.txt'.format(str(i)), pc, fmt="%.8f")
