# System libs
import os
import time
import argparse
import datetime as datetime
import random

# Numerical libs
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import inspect
# Our libs
from dataset.DrivAerNet import PRESSURE_MEAN, PRESSURE_STD
from dataset.dataset import *
from model.SlotAttnPointTransformer_MultiFlowCondAdapter import PointTransformer_MultiFlowCondAdapter as Model
from utils_train import AverageMeter, get_params_groups, cosine_scheduler, MultiEpochsDataLoader, load_model_weights
torch.set_float32_matmul_precision('high')
seed_value = 200099  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）


def loss_with_checking(pred: torch.Tensor,
                       target: torch.Tensor,
                       loss):
    """
    仅在 pred/target 为有限数的位置计算损失。
    - 支持函数式（如 F.mse_loss）与模块式（如 nn.MSELoss）loss。
    - 最终返回掩码后的均值损失。
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"

    # 有效掩码：默认仅检查 target；若你担心 pred 也可能出现 NaN/Inf，可一起检查
    mask = torch.isfinite(target)
    # 可选：同时检查 pred（更稳健）
    mask = mask & torch.isfinite(pred)
    if not mask.any():
        # 没有任何有效元素：返回与 pred 相连的零（不会破坏图）
        return (pred * 0.0).sum()

    pred = pred[mask==True]
    target = target[mask==True]
    return loss(pred, target)

# train one epoch
def checkpoint(nets, optimizer, args, epoch):
    # print('Saving checkpoints...')
    net_encoder = nets.module
    
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model.pth'.format(args.saveroot))
    
def train(model, data_loader, optimizers, epoch, gpu, lr_schedule, scaler, args):
    batch_time = AverageMeter()
    ave_loss_1 = AverageMeter()
    ave_loss_2 = AverageMeter()
    ave_loss_3 = AverageMeter()
    ave_loss_4 = AverageMeter()
    ave_loss_5 = AverageMeter()
    
    model.train()
    epoch_iters = len(data_loader)
    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    for idx,data in enumerate(data_loader):
        
        it = len(data_loader) * epoch + idx
        for i, param_group in enumerate(optimizers.param_groups):
            param_group["lr"] = lr_schedule[it] * param_group["base_lr"]

        point_cloud, pressure, cond, route, Cd = data
        point_cloud = point_cloud.float().cuda(gpu)
        pressure = pressure.float().cuda(gpu)
        cond = cond.float().cuda(gpu)
        Cd = Cd.float().cuda(gpu)
        route = route.cuda(gpu)
        
        optimizers.zero_grad()
        # tau = tau_schedule[it]
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            rec = model(point_cloud[:,:,:3], cond, route)
             
            loss_rec = F.l1_loss(rec[:,:,0], pressure)
            loss_total = loss_rec
        
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizers)
        scaler.update()

        # # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # # update average loss and acc
        ave_loss_1.update(loss_rec)
        # ave_loss_2.update(loss_cd)

        if dist.get_rank()==0 and idx % 10 == 0:
            f = open('log/{}.txt'.format(args.saveroot.split('/')[-1]), 'a')
            print('[{}][{}/{}], lr: {:.2f}, '
                  'time: {:.2f}, '
                  'Loss: {:.6f}'
                  .format(epoch, idx, epoch_iters, lr_schedule[it], batch_time.average(),
                  ave_loss_1.average()), file=f)
            f.close()


    # torch.save(
    #     optimizer.state_dict(),
    #     '{}/opt_epoch_{}.pth'.format(args.saveroot, epoch))


def main(gpu,args):
    # Network Builders
    load_gpu = gpu+args.start_gpu
    rank = gpu
    torch.cuda.set_device(load_gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=args.gpu_num,
        rank=rank,
        timeout=datetime.timedelta(seconds=300))

    if args.dataset_option == 0:
        dataset_train = point_cloud_dataset(num_points=8192)
    elif args.dataset_option == 1:
        dataset_train = point_cloud_dataset_no_DrivAerNet(num_points=8192)
    elif args.dataset_option == 2:
        dataset_train = point_cloud_dataset_no_train(num_points=8192)
    elif args.dataset_option == 3:
        dataset_train = point_cloud_dataset_no_Wings(num_points=8192)
    elif args.dataset_option == 4:
        dataset_train = point_cloud_dataset_no_NGAD(num_points=8192)
    elif args.dataset_option == 5:
        dataset_train = point_cloud_dataset_no_FlowBench(num_points=8192)
    print(dataset_train.__len__())
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, seed=seed_value)
    loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, sampler=sampler_train, 
                                    pin_memory=True, num_workers=args.workers, drop_last=True)
    
    # loader_train = get_dataloaders(
    #     dataset_path='/data/home/zdhs0017/zoujunhong/data/DrivAerNet++/Pressure',
    #     num_points=args.points,
    #     batch_size=args.batchsize,
    #     world_size=args.gpu_num,
    #     rank=gpu,
    #     num_workers=args.workers)
    # load nets into gpu
    model = Model(
        depth=[8,8,8,8,8],
        channels=[192,384,768,1536,3072],
        num_points=[1024, 256, 64, 16],
        k=16
    )
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('模型参数量：{}M'.format(count_parameters(model)/1e6))
    
    
    if args.checkpoint != '':
        load_model_weights(model, args.checkpoint)
    
    if args.resume_epoch!=0:
        to_load = torch.load(os.path.join(args.saveroot,'model_epoch_{}.pth'.format(args.resume_epoch)),map_location=torch.device("cpu"))
        model.load_state_dict(to_load,strict=True)
        
    model = model.cuda(load_gpu).float()

    model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[load_gpu],
                find_unused_parameters=False)

    model = torch.compile(model)
    # Set up optimizers
    param_groups = get_params_groups(model, lr = args.lr, wd=0.01, special_keyword=['adapt'], special_lr=[2.5*args.lr])
    optimizer = torch.optim.AdamW(param_groups, eps=1e-7)
    
    lr_schedule = cosine_scheduler(
        1.00,  # linear scaling rule
        0.01,
        args.total_epoch, 
        len(loader_train),
        warmup_iters=3000)

    
    # Main loop
    scaler = torch.amp.GradScaler(enabled=True)
    
    for epoch in range(args.resume_epoch, args.total_epoch):
        # print('Epoch {}'.format(epoch))
        train(model, loader_train, optimizer, epoch, load_gpu, lr_schedule, scaler, args)

        # checkpointing
        if dist.get_rank() == 0 and (epoch+1)%args.save_step==0:
            checkpoint(model, optimizer, args, epoch+1)

    print('Training Done!')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument("--batchsize",type=int,default=2)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=4)
    parser.add_argument("--lr",type=float,default=4e-5)
    parser.add_argument("--saveroot",type=str,default='/data/home/zdhs0017/zoujunhong/model/savemodel/field/SAPT_Unified_pretrain_2b_p8192')
    # parser.add_argument("--checkpoint",type=str,default='/data/home/zdhs0017/zoujunhong/model/savemodel/field/SAPT_Unified_pretrain_1b_p8192/model.pth')
    parser.add_argument("--checkpoint",type=str,default='/data/home/zdhs0017/zoujunhong/model/savemodel/field/SAPT_DrivAerNet_scratch_pressure+coeff_2b_p8192/model.pth')
    # parser.add_argument("--checkpoint",type=str,default='')
    parser.add_argument("--total_epoch",type=int,default=30)
    parser.add_argument("--resume_epoch",type=int,default=0)
    parser.add_argument("--save_step",type=int,default=5)
    parser.add_argument("--dataset_option",type=int,default=0)
    parser.add_argument("--points",type=int,default=8192)
    
    parser.add_argument("--port",type=int,default=45325)
    args = parser.parse_args()

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(args,))
