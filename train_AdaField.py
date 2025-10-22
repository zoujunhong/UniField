# System libs
import os
import time
import argparse
import datetime as datetime
import random

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
# Our libs
from dataset import *
from model.AdaField import PointTransformer_CondAdapter as Model
from utils_train import AverageMeter, get_params_groups, cosine_scheduler, load_model_weights
torch.set_float32_matmul_precision('high')
seed_value = 200099  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

# train one epoch
def checkpoint(nets, optimizer, args, epoch):
    # print('Saving checkpoints...')
    net_encoder = nets.module
    
    if not os.path.exists(args.saveroot):
        os.makedirs(args.saveroot, exist_ok=True)
        
    torch.save(
        net_encoder.state_dict(),
        '{}/model.pth'.format(args.saveroot, epoch))
    
def train(model, data_loader, optimizers, epoch, gpu, lr_schedule, scaler, args):
    batch_time = AverageMeter()
    ave_loss_1 = AverageMeter()
    
    model.train()
    epoch_iters = len(data_loader)
    data_loader.sampler.set_epoch(epoch)

    # main loop
    tic = time.time()
    for idx,data in enumerate(data_loader):
        
        it = len(data_loader) * epoch + idx
        for i, param_group in enumerate(optimizers.param_groups):
            param_group["lr"] = lr_schedule[it] * param_group["base_lr"]

        point_cloud, pressure, cond, route = data
        point_cloud = point_cloud.float().cuda(gpu)
        pressure = pressure.float().cuda(gpu)
        cond = cond.float().cuda(gpu)
        route = route.long().cuda(gpu)
        
        optimizers.zero_grad()
        # tau = tau_schedule[it]
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            rec = model(point_cloud[:,:,:3], cond)
            loss_rec = F.l1_loss(rec[:,:,0], pressure)
            loss_total = loss_rec
        
        # # Backward
        
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
        # ave_loss_2.update(loss_coeff)

        if dist.get_rank()==0 and idx % 10 == 0:
            
            f = open('log/{}.txt'.format(args.saveroot.split('/')[-1]), 'a')
            print('[{}][{}/{}], lr: {:.2f}, '
                  'time: {:.2f}, '
                  'Loss: {:.6f}'
                  .format(epoch, idx, epoch_iters, lr_schedule[it], batch_time.average(),
                  ave_loss_1.average()), file=f)
            f.close()


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

    if args.dataset == 'flowbench':
        dataset_train = FlowBench_3D_LDC(data_root=args.data_root, repeat=10)
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, seed=seed_value)
        loader_train = MultiEpochsDataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, sampler=sampler_train, 
                                        pin_memory=True, num_workers=args.workers)
    elif args.dataset == 'drivaernet++':
        loader_train = get_dataloaders(
            dataset_path=args.data_root,
            num_points=args.points,
            batch_size=args.batchsize,
            world_size=args.gpu_num,
            rank=gpu,
            num_workers=args.workers)
    
    # load nets into gpu
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
    param_groups = get_params_groups(model, lr = args.lr, wd=0.01, special_keyword=['adapt'], special_lr=[args.lr])
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
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--start_gpu",type=int,default=0)
    parser.add_argument("--gpu_num",type=int,default=4)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--saveroot",type=str,default='/path/to/cheakpoint_saveroot')
    parser.add_argument("--checkpoint",type=str,default='')
    parser.add_argument("--total_epoch",type=int,default=100)
    parser.add_argument("--resume_epoch",type=int,default=0)
    parser.add_argument("--save_step",type=int,default=20)
    
    parser.add_argument('--dataset', type=str, default="drivaernet++", choices=["drivaernet++", "flowbench"]) 
    parser.add_argument('--data_root', type=str, default="/path/to/dataset/root_dir") 
    # /path/to/DrivAerNet++/Pressure for drivaernet++ and /path/to/FlowBench/LDC_NS_3D/point_cloud/ for flowbench
    
    parser.add_argument("--points",type=int,default=8192)
    parser.add_argument("--modelscale", type=str, default="250m", choices=["250m", "1b", "2b", "customize"])
    # for "customize" choice, specify the network hyperparameter at line 150
    
    parser.add_argument("--port",type=int,default=45325)
    args = parser.parse_args()

    print(args)

    mp.spawn(main, nprocs=args.gpu_num, args=(args,))
