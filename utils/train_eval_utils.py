import sys
import torch
import pytz
import os
import shutil
import torch_harmonics as th
import torch.cuda.amp as amp

from tqdm import tqdm
from utils.distributed_utils import reduce_value, is_main_process
from matplotlib import pyplot as plt
from torch import autograd
from datetime import datetime
from tqdm import tqdm


beijing_tz = pytz.timezone('Asia/Shanghai')


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



