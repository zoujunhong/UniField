from .PDEBench import *
from .FlowBench import *
from .train_data import *
from .Ahmed import *
from .DrivAerNet import *
from .Flight import *
from .Nga import *
from .Magnetic import *
from .Ahmed import *
from .DrivAerML import *
from torch.utils.data import ConcatDataset

def point_cloud_dataset(num_points=10000):
    return ConcatDataset([
        train_dataset(num_points=num_points),
        get_datasets(dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure', num_points=num_points),
        Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', num_points, train_samples=1260, repeat=1),
        NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points, verbose=False),
        # DrivAerML(repeat=10, num_points=num_points),
        FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=3, num_points=num_points)
        ])

def point_cloud_dataset_no_train(num_points=10000):
    return ConcatDataset([
        get_datasets(dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure', num_points=num_points),
        Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', num_points, train_samples=1260, repeat=1),
        NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points, verbose=False),
        FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=3, num_points=num_points)
        ])

def point_cloud_dataset_no_DrivAerNet(num_points=10000):
    return ConcatDataset([
        train_dataset(num_points=num_points),
        Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', num_points, train_samples=1260, repeat=1),
        NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points, verbose=False),
        FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=3, num_points=num_points)
        ])

def point_cloud_dataset_no_Wings(num_points=10000):
    return ConcatDataset([
        train_dataset(num_points=num_points),
        get_datasets(dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure', num_points=num_points),
        NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points, verbose=False),
        FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=3, num_points=num_points)
        ])

def point_cloud_dataset_no_NGAD(num_points=10000):
    return ConcatDataset([
        train_dataset(num_points=num_points),
        get_datasets(dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure', num_points=num_points),
        Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', num_points, train_samples=1260, repeat=1),
        FlowBench_3D_LDC(data_root='/data/home/zdhs0017/zoujunhong/data/FlowBench/LDC_NS_3D/point_cloud/', repeat=3, num_points=num_points)
        ])

def point_cloud_dataset_no_FlowBench(num_points=10000):
    return ConcatDataset([
        train_dataset(num_points=num_points),
        get_datasets(dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure', num_points=num_points),
        Wings3D('/data/home/zdhs0017/zoujunhong/data/Flight/3D_Wings_full/aircraft', num_points, train_samples=1260, repeat=1),
        NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points, verbose=False),
        ])

def train_dataset(num_points):
    return ConcatDataset([
    Traindata_3D_linear(num_points=num_points, repeat=50, train_samples=36),
    Mag(num_points=num_points, repeat=100, train_samples=20)
    ])
