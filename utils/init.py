import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy
import torch
import os
import argparse
import xarray as xr
import argparse
import sys
import emoji
import re
import shutil
import yaml
# import weatherbench2

from matplotlib import pyplot as plt

sys.path.append(".")

from tqdm import tqdm


# 规定MERRA-2变量
def get_vars():

    # vars_dict = {
    #     "tavg1_2d_aer_Nx": ['BCEXTTAU', 'BCSMASS', 'DUEXTTAU', 'DUSMASS', 'OCEXTTAU',
    #                         'OCSMASS', 'SO4SMASS', 'SSEXTTAU', 'SSSMASS', 'SUEXTTAU',
    #                         'TOTEXTTAU', 'TOTSCATAU'],
    #     "tavg1_2d_flx_Nx": ['QLML', 'TLML', 'ULML', 'VLML', 'PBLH', 'PRECTOT'],
    #     "tavg3_3d_asm_Nv": {"var": ["QV", "SLP", "T", "U", "V"],
    #                         "level": [45, 48, 51, 53, 56, 60, 63, 68, 72]}
    # }

    # # V0
    # vars_dict = {
    #     "tavg1_2d_aer_Nx": ['DUEXTTAU', 'DUSMASS', 'TOTEXTTAU'],
    #     "tavg1_2d_flx_Nx": ['QLML', 'TLML', 'ULML', 'VLML', 'PBLH', 'PRECTOT'],
    #     "tavg3_3d_asm_Nv": {"var": ["QV", "SLP", "T", "U", "V"],
    #                         "level": [45, 48, 51, 53, 56, 60, 63, 68, 72]}
    # }

    # V1
    vars_dict = {
        "tavg1_2d_aer_Nx": ['BCEXTTAU', 'BCSMASS', 'DUEXTTAU', 'DUSMASS', 'OCEXTTAU',
                            'OCSMASS', 'SO4SMASS', 'SSEXTTAU', 'SSSMASS', 'SUEXTTAU',
                            'TOTEXTTAU', 'TOTSCATAU'],
        "tavg1_2d_flx_Nx": ['QLML', 'TLML', 'ULML', 'VLML', 'PRECTOT'],
        "tavg3_3d_asm_Nv": {"var": ["QV", "SLP", "T", "U", "V"],
                            "level": [45, 48, 51, 53, 56, 60, 63, 68, 72]}
    }

    return vars_dict


# 初始化三小时模型（改为加载 JIT traced 模型）
def load_3h_model(device):
    print(emoji.emojize(':black_circle: loading 3h model'))
    # 修改为加载 JIT traced 模型
    model_3h = torch.jit.load("./model/gamfs_3h_traced.pt", map_location=device)
    model_3h = model_3h.to(device)
    model_3h.eval()  

    return model_3h

# 初始化6小时模型（改为加载 JIT traced 模型）
def load_6h_model(device):
    print(emoji.emojize(':black_circle: loading 6h model'))
    # 修改为加载 JIT traced 模型
    model_6h = torch.jit.load("./model/gamfs_6h_traced.pt", map_location=device)
    model_6h = model_6h.to(device)
    model_6h.eval()  

    return model_6h

# 初始化9小时模型（改为加载 JIT traced 模型）
def load_9h_model(device):
    """
    导入9小时预报模型
    @param device: 导入设备
    @return: eval状态下的模型
    """
    print(emoji.emojize(':black_circle: loading 9h model'))
    # 修改为加载 JIT traced 模型
    model_9h = torch.jit.load("./model/gamfs_9h_traced.pt", map_location=device)
    model_9h = model_9h.to(device)
    model_9h.eval()  

    return model_9h

# 初始化12小时模型（改为加载 JIT traced 模型）
def load_12h_model(device):
    """
    导入12小时预报模型
    @param device: 导入设备
    @return: eval状态下的模型
    """
    print(emoji.emojize(':black_circle: loading 12h model'))
    # 修改为加载 JIT traced 模型
    model_12h = torch.jit.load("./model/gamfs_12h_traced.pt", map_location=device)
    model_12h = model_12h.to(device)
    model_12h.eval() 

    return model_12h


if __name__ == "__main__":

    model = load_12h_model("TransUnet", "cuda:0")
    torch.save(model, "/data/_project_zxt/merra_forecast/checkpoint/gamfs_models/gamfs_12h.pth")




