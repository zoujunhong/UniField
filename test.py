import argparse
# Numerical libs
import torch
import torch.nn.functional as F
# Our libs
from dataset import *
from utils_train import load_model_weights

import datetime as datetime

    
def test(args, model, data_loader):
    model.eval()
    total_l2re = 0.
    total_l1re = 0.
    total_mse = 0.
    total_mae = 0.
    total_maxerror = 0.
    total_rmse = 0.
    count = 0.
    
    torch.set_printoptions(precision=4,sci_mode=False,linewidth=1000)
    for idx,data in enumerate(data_loader):
        if idx % 100 == 0:
            print(idx)
        with torch.no_grad():
            # while True:
                point_cloud, pressure, cond, route = data
                point_cloud = point_cloud.float().cuda()
                pressure = pressure.float()
                cond = cond.float().cuda()
                route = route.cuda()
                
                point_cloud = point_cloud.reshape(args.points//args.model_points,args.model_points,3)
                cond = cond.repeat(args.points//args.model_points,1)
                route = route.repeat(args.points//args.model_points)
                # 输入到模型处理
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    rec = model(point_cloud, cond, route)  # 假设输出为 [B * num_chunks, ...]
                    rec = rec.reshape(1, args.points)

                L2RE = torch.sum((pressure - rec) ** 2)**0.5 / torch.sum(pressure ** 2)**0.5
                L1RE = torch.sum(torch.abs(pressure - rec)) / torch.sum(torch.abs(pressure))
                # print(pred_cd.shape, cd.shape)
                mse = F.mse_loss(rec, pressure)
                mae = F.l1_loss(rec, pressure)
                rmse = mse**0.5
                
                total_mse += mse
                total_mae += mae
                total_maxerror = max(total_maxerror, mae.item())
                total_l2re += L2RE
                total_l1re += L1RE
                total_rmse += rmse
                count += 1
                
                print('MSE(x1e-2): {}, MAE(x1e-1): {}, MaxAE: {}, RelL2(%): {}, RelL1(%): {}'\
                    .format(100*total_mse.item()/count, 10*total_mae.item()/count, total_maxerror, 100*total_l2re/count, 100*total_l1re/count))


                
    

def main(args):
    # Network Builders
    if args.modeltype == 'UniField':
        from model.UniField import PointTransformer_MultiFlowCondAdapter as Model
    elif args.modeltype == 'AdaField':
        from model.AdaField import PointTransformer_CondAdapter as Model
    
    if args.modelscale == '250m':
        model = Model(
            depth=[4, 4, 6, 12, 8],
            channels=[64, 128, 256, 512, 1024],
            num_points=[1024, 256, 64, 16],
            out_channels=1,
            cond_dims=2,
            k=16)
    elif args.modelscale == '1b':
        model = Model(
            depth=[4, 4, 6, 12, 8],
            channels=[128, 256, 612, 1024, 2048],
            num_points=[1024, 256, 64, 16],
            out_channels=1,
            cond_dims=2,
            k=16)
    elif args.modelscale == '2b':
        model = Model(
            depth=[8, 8, 8, 8, 8],
            channels=[192, 384, 768, 1536, 3072],
            num_points=[1024, 256, 64, 16],
            out_channels=1,
            cond_dims=2,
            k=16)
    elif args.modelscale == 'customize': # you can also customize your model parameter here
        model = Model(
            depth=[8, 8, 8, 8, 8],
            channels=[192, 384, 768, 1536, 3072],
            num_points=[1024, 256, 64, 16],
            out_channels=1,
            cond_dims=2,
            num_routes=3,
            k=16)
    
    # load nets into gpu
    load_model_weights(model, args.checkpoint_path)
    model = model.cuda()
    
    if args.dataset == 'flowbench':
        dataset = FlowBench_3D_LDC(data_root=args.data_root, repeat=1, num_points=args.points, split='val')
        loader_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False,
                                        num_workers=4)
    elif args.dataset == 'drivaernet':
        loader_train = get_val_dataloaders(
            dataset_path=args.data_root,
            num_points=args.points,
            batch_size=1,
            num_workers=4)
    
    test(args, model, loader_train)


    print('Testing Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("--modeltype", type=str, default="UniField", choices=["AdaField", "UniField"])
    parser.add_argument("--modelscale", type=str, default="250m", choices=["250m", "1b", "2b", "customize"])
    parser.add_argument("--checkpoint_path",type=str,default='')
    
    parser.add_argument("--dataset", type=str, default="drivaernet", choices=["drivaernet", "flowbench"])
    parser.add_argument('--data_root', type=str, default="/path/to/DrivAerNet++/Pressure") 
    parser.add_argument('--route',type=int,default=0) 
    # the route number of dataset, keep consistent with training
    
    parser.add_argument("--points",type=int,default=8192) # The number of sampled points in each sample
    parser.add_argument("--model_points",type=int,default=8192) # The number of points processed by the model in a single run

    args = parser.parse_args()

    assert args.points % args.model_points == 0, 'The number of points in each sample needs to be divisible by the number of points processed by the model in a single run.'
    print(args)

    main(args)
