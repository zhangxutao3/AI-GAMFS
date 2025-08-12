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
import re
import shutil
import warnings

from matplotlib import pyplot as plt

sys.path.append(".")
warnings.filterwarnings("ignore")


from tqdm import tqdm
from utils.init import get_vars
from matplotlib import pyplot as plt
from datetime import datetime



# 转换数据格式
# 整合nc数据
def convert2dataarray(info, data_array, variables, time_range, ground_lat, ground_lon):

    # 初始化DataArray字典
    data_vars = {}
    BASE_var = []

    # 遍历变量，根据是否有下划线来区分处理
    for i, var in enumerate(variables):
        attrs = info[i]
        if '_' in var:
            base_var, level = var.split('_')
            level = int(level)  # 转换为整数
            if base_var not in data_vars:
                levels = set(
                    int(v.split('_')[1]) for v in variables if v.startswith(base_var) and '_' in v
                )
                data_vars[base_var] = xr.DataArray(
                    np.zeros((len(time_range),
                              len(levels),
                              len(ground_lat), len(ground_lon))),
                    dims=('time', 'level', 'latitude', 'longitude'),
                    coords={'time': time_range,
                            'level': sorted(set(int(v.split('_')[1]) for v in variables if v.startswith(base_var) and '_' in v)),
                            'latitude': ground_lat, 'longitude': ground_lon},
                    attrs=attrs
                )
            # 填充对应的level数据
            level_index = list(data_vars[base_var].coords['level'].values).index(level)
            data_vars[base_var][:, level_index, :, :] = data_array[:, i, :, :]
            BASE_var.append(base_var)
        else:
            # 对于没有下划线的变量，直接保存
            data_vars[var] = xr.DataArray(
                data_array[:, i, :, :],
                dims=('time', 'latitude', 'longitude'),
                coords={'time': time_range, 'latitude': ground_lat, 'longitude': ground_lon},
                attrs=attrs
            )
            BASE_var.append(var)

    # 创建一个Dataset
    dataset = xr.Dataset(data_vars)

    return dataset, list(dict.fromkeys(BASE_var))



def closest_hour_marker(time_str):
    """
    判断给定时间字符串的小时部分离 00 时刻更近还是离 12 时刻更近

    :param time_str: 时间字符串，格式为 "YYYY-MM-DD HH:MM"
    :return: "00H" 如果离 00 时刻更近，否则 "12H"
    """
    # 解析时间字符串为datetime对象
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

    # 提取小时部分
    hour = time.hour

    # 计算小时部分与 00 点和 12 点的距离
    distance_to_00 = min(hour, 24 - hour)  # 考虑到 24 小时制的循环
    distance_to_12 = abs(hour - 12)

    # 比较两个距离，返回相应的结果
    if distance_to_00 <= distance_to_12:
        return "00H"
    else:
        return "12H"




# 标准化
def stand(index: list, data: np.ndarray, scale_factor: float = 1.0):

    norm = pd.read_csv("./merra_norm.csv")
    for i in range(len(index)):
        x = norm[norm["index"] == index[i]]
        mean_ = x["mean"].values
        std_ = x["std"].values
        # 标准化后乘以scale_factor
        data[i, :, :] = ((data[i, :, :] - mean_) / std_) * scale_factor

    return data


# 反标准化
def anti_stand(index: list, data: np.ndarray, scale_factor: float = 1.0):

    norm = pd.read_csv("./merra_norm.csv")
    for i in range(len(index)):
        x = norm[norm["index"] == index[i]]
        mean_ = x["mean"].values
        std_ = x["std"].values
        # 反标准化前除以scale_factor
        data[i, :, :] = (data[i, :, :] / scale_factor) * std_ + mean_

    return data



# 导入模式实况初始场
def open_asm_merra(time, scale_factor = 2, interp_method ="linear"):

    '''
    用于读入GEOS同化初始场
    @param time: 时间
    @param scale_factor: 初始场下采样比例
    @param interp_method: 下采样方法
    @return: 数据矩阵
    '''

    var_dict = get_vars()

    # 获取初始场日期字符串
    time_str = time.replace(" ", "_")
    time_str = time_str.replace("-", "")
    time_str = time_str.replace(":", "")

    # 输出结果
    total_matrix = []
    index = []

    # 获取数据
    for key, value in var_dict.items():

        # tavg1_2d_aer_Nx (tavg3_2d_aer_Nx)
        if key == "tavg1_2d_aer_Nx":

            folder_path = "./temp_asm/GEOS.fp.asm." + key.replace("tavg1", "tavg3") + "/"
            file_list = glob.glob(folder_path + "*.hdf")
            file = [file for file in file_list if time_str in file]

            if len(file) == 1:

                file = file[0]
                f = xr.open_dataset(file)
                for var in value:
                    data = f[var].loc[time, :, :]
                    data = data.interp(lon=data.lon.values[::scale_factor], lat=data.lat.values[::scale_factor], method=interp_method)
                    total_matrix.append(data.values.squeeze())
                    index.append(var)
                f.close()

        # tavg1_2d_flx_Nx
        elif key == "tavg1_2d_flx_Nx":

            folder_path = "./temp_asm/GEOS.fp.asm." + key + "/"
            file_list = glob.glob(folder_path + "*.hdf")
            file = [file for file in file_list if time_str in file]

            if len(file) == 1:

                file = file[0]
                f = xr.open_dataset(file)
                for var in value:
                    data = f[var].loc[time, :, :]
                    data = data.interp(lon=data.lon.values[::scale_factor], lat=data.lat.values[::scale_factor], method=interp_method)
                    total_matrix.append(data.values.squeeze())
                    index.append(var)
                f.close()

        # tavg3_3d_asm_Nv
        elif key == "tavg3_3d_asm_Nv":

            folder_path = "./temp_asm/GEOS.fp.asm." + key + "/"
            file_list = glob.glob(folder_path + "*.hdf")
            file = [file for file in file_list if time_str in file]

            if len(file) == 1:

                file = file[0]
                f = xr.open_dataset(file)

                for var_ in value["var"]:
                    if var_ in ["H", "QV", "RH", "T", "U", "V", "OMEGA"]:
                        for level_ in value["level"]:
                            data = f[var_].loc[time, level_, :, :]
                            data = data.interp(lon=data.lon.values[::scale_factor], lat=data.lat.values[::scale_factor], method=interp_method)
                            total_matrix.append(data.values.squeeze())
                            index.append(var_ + "_" + str(level_))
                    else:
                        data = f[var_].loc[time, :, :]
                        data = data.interp(lon=data.lon.values[::scale_factor], lat=data.lat.values[::scale_factor], method=interp_method)
                        total_matrix.append(data.values.squeeze())
                        index.append(var_)
                f.close()

    total_matrix = np.stack(total_matrix, axis=0)

    lon = data["lon"].values
    lat = data["lat"].values

    if total_matrix.shape[0] == 54:
        return total_matrix, index, lon, lat, time_str




