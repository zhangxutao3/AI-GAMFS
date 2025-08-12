import os
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import datetime
import sys
import pytz
import shutil
import torch.cuda.amp as amp
import warnings
import argparse
import datetime as d

from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import autograd
from datetime import datetime


"""
模型训练相关代码，重新整理，以便模型训练
"""

beijing_tz = pytz.timezone('Asia/Shanghai')
warnings.filterwarnings("ignore")


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, timeout=d.timedelta(seconds=3600*6))
    dist.barrier()



def cleanup():
    dist.destroy_process_group()




def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0



def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def train(model, optimizer, data_loader, device, epoch, loss_function, global_train_loss):

    model.train()
    loss_function = loss_function
    mean_loss = torch.zeros(1).to(device)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        optimizer.zero_grad()

        x, y, time = data
        x, y, time = x.to(device, non_blocking=True), y.to(device, non_blocking=True), time.to(device, non_blocking=True)
        x, y, time = x.to(torch.float32), y.to(torch.float32), time.to(torch.float32)

        y_hat = model(x, time=time)

        loss = loss_function(y_hat, y)
        loss.backward()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[{}] [epoch:{} lr:{}] train mean loss {} | {}".format(
                 datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S"),
                 epoch,
                 round(optimizer.state_dict()['param_groups'][0]['lr'], 7),
                 mean_loss.item(),
                 global_train_loss)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function, global_test_loss):

    model.eval()

    # 平均损失
    mean_loss = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        x, y, time = data
        x, y, time = x.to(device, non_blocking=True), y.to(device, non_blocking=True), time.to(device, non_blocking=True)
        x, y, time = x.to(torch.float32), y.to(torch.float32), time.to(torch.float32)

        y_hat = model(x, time=time)

        loss = loss_function(y_hat, y)

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[{}] [epoch:{} lr:None] test mean loss {} | {}".format(
               datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S"),
               epoch,
               mean_loss.item(),
               global_test_loss)


    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


def save_loss_to_file(epoch, train_loss, test_loss, lr, file_path='losses.txt'):
    # 当index和epoch都为0时，如果文件已存在，则删除
    if epoch == 0 and os.path.exists(file_path):
        os.remove(file_path)

    # 以追加模式打开文件，如果文件不存在，将会创建
    with open(file_path, 'a') as file:
        # 写入当前的epoch, index, train loss, test loss和学习率
        file.write(f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, LR: {lr}\n")



def flatten_dict(d, parent_key='', sep='_'):
    """ 扁平化字典 """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):  # 如果是嵌套字典，则递归扁平化
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0




def training_log(log_path, save_path):

    """
    自动绘制训练的损失曲线
    """

    with open(log_path, 'r') as file:
        lines = file.readlines()

    # 步骤2: 解析数据
    epochs = []
    train_losses = []
    test_losses = []
    lrs = []

    for line in lines:
        if "Epoch" in line:
            parts = line.split(',')
            epoch = int(parts[0].split(':')[1].strip())
            train_loss = float(parts[1].split(':')[1].strip())
            test_loss = float(parts[2].split(':')[1].strip())
            lr = float(parts[3].split(':')[1].strip())

            epochs.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            lrs.append(lr)

    # 步骤3: 绘图
    fig, ax1 = plt.subplots()

    x = np.arange(0, len(train_losses), 1)

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    # ax1.plot(x, train_losses, label='Train Loss', color='r')
    ax1.plot(x, test_losses, label='Test Loss', color='g', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    plt.grid()

    ax2 = ax1.twinx()  # 实例化一个双y轴
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)  # 我们已经处理了ax1的xlabel和ylabel
    ax2.plot(x, lrs, label='Learning Rate', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # 调整整体空白
    plt.savefig(save_path, dpi=500)







if __name__ == '__main__':

    train_dict = {
        'epochs': 130,
        'batch_size': 3,
        'lr': 3e-4,
        'lrf': 1e-3,
        'stopping_patience': 10,
        'use_checkpoint': True,
        'dataloader_nw': 3,
        'prefetch_factor': 2,
        'syncBN': False,
        'log_path': "/data/_project_zxt/merra_forecast/log/log_6h.txt",
        'device': 'cuda',
        'world_size': 4,
        'dist_url': 'env://',
        'checkpoint_path': "/data/_project_zxt/merra_forecast/checkpoint/",
    }

    model_dict = {
        'input_dim': (361, 576),
        'img_dim': (576, 576),
        'in_channels': 54,
        'vit_blocks': 14,
        'vit_heads': 16,
        'inplanes': 640,
        'patch_size': 2,
        'vit_transformer_dim': 1200 * 2,
        'vit_dim_linear_mhsa_block': 1200 * 2,  # transformer内部dim
        'weights_path': "/data/_project_zxt/merra_forecast/weights/MERRA-model-0.pth",
    }

    handle = model_function(
        train_dict,
        model_dict
    )





