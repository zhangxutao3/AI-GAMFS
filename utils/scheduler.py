import pandas as pd
import torch
import sys

# sys.path.append("/data/_project_zxt/merra_forecast/")

from utils.handle import stand, anti_stand


def drop_negative(index, data, device):


    drop_vars = [
        'BCEXTTAU', 'BCSMASS', 'DUEXTTAU', 'DUSMASS', 'OCEXTTAU',
        'OCSMASS', 'SO4SMASS', 'SSEXTTAU', 'SSSMASS', 'SUEXTTAU',
        'TOTEXTTAU', 'TOTSCATAU', 'QLML', 'PRECTOT', 'QV_45', 'QV_48',
        'QV_51', 'QV_53', 'QV_56', 'QV_60', 'QV_63', 'QV_68', 'QV_72',
        'SLP'
    ]

    # 创建数据的副本以避免修改原始数据
    data = data.clone()
    data = data.detach().cpu().numpy().squeeze().copy()

    # 反标准化
    data = anti_stand(index, data)

    # 根据 index 找到对应的变量，并对 drop_vars 中的变量进行处理
    for var in drop_vars:
        if var in index:  # 检查变量是否在 index 中
            var_index = index.index(var)  # 获取该变量在矩阵中的索引
            data[var_index][data[var_index] < 0] = 0  # 将负值抹去（设置为 0）

    # 重新标准化
    data = stand(index, data)

    return torch.tensor(data).unsqueeze(0).to(device)



# AI-GAFS滚动预报
@torch.no_grad()
def rolling_model(init_field,
                  target_step,
                  index,
                  model_3h,
                  model_6h,
                  model_9h,
                  model_12h,
                  device,
                  method="3+6",
                  start_time=None):
    """
    滚动训练核心代码（卧槽写的真牛逼）
    @param init_field: 初始场
    @param target_step: 目标步长
    @param index: 索引
    @param model_3h: 3小时模型
    @param model_6h: 6小时模型
    @param model_9h: 9小时模型
    @param model_12h: 12小时模型
    @param device: 设备
    @param method: 方法 optional: [3, 3+6, 3+6+9, 3+6+9+12]
    @param start_time: 开始时刻
    @return: 指定目标的预报值(已反归一化)
    """

    # 跳过初始场时刻 (不需要加3小时)
    time = start_time

    if method == "3+6":

        # 计算6小时和3小时模型的使用次数
        num_6h = target_step // 2  # 每两个时间步（6小时）用一次6小时模型
        num_3h = target_step % 2  # 剩余的时间步（3小时）用一次3小时模型
        # 根据初始场
        output = init_field
        # 先运行6小时模型
        for _ in range(num_6h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_6h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=6)
        # 然后运行3小时模型
        if num_3h > 0:
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_3h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=3)

    elif method == "3+6+12":

        # 计算12小时、6小时和3小时模型的使用次数
        num_12h = target_step // 4  # 每四个时间步（12小时）用一次12小时模型
        remaining_hours = target_step % 4  # 剩余的时间步
        num_6h = remaining_hours // 2  # 剩余时间步中每两个时间步（6小时）用一次6小时模型
        num_3h = remaining_hours % 2  # 剩余时间步中每一个时间步（3小时）用一次3小时模型

        # 根据初始场
        output = init_field

        # 先运行12小时模型
        for _ in range(num_12h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_12h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=12)

        # 然后运行6小时模型
        for _ in range(num_6h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_6h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=6)

        # 最后运行3小时模型
        if num_3h > 0:
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_3h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=3)

    elif method == "3":

        # 获取初始场
        output = init_field
        # 滚动迭代
        for _ in range(target_step):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_3h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=3)


    elif method == "3+6+9":
        # 计算9小时、6小时和3小时模型的使用次数
        num_9h = target_step // 3  # 每三个时间步（9小时）用一次9小时模型
        remaining_hours = target_step % 3  # 剩余的时间步
        num_6h = remaining_hours // 2  # 剩余时间步中每两个时间步（6小时）用一次6小时模型
        num_3h = remaining_hours % 2  # 剩余时间步中每一个时间步（3小时）用一次3小时模型

        output = init_field

        # 运行9小时模型
        for _ in range(num_9h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_9h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=9)

        # 运行6小时模型
        for _ in range(num_6h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_6h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=6)

        # 运行3小时模型
        if num_3h > 0:
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_3h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=3)


    elif method == "3+6+9+12":
        # 计算12小时、9小时、6小时和3小时模型的使用次数
        num_12h = target_step // 4  # 每四个时间步（12小时）用一次12小时模型
        remaining_hours = target_step % 4  # 剩余的时间步
        num_9h = remaining_hours // 3  # 每三个时间步（9小时）用一次9小时模型
        remaining_hours %= 3
        num_6h = remaining_hours // 2  # 每两个时间步（6小时）用一次6小时模型
        num_3h = remaining_hours % 2  # 每一个时间步（3小时）用一次3小时模型

        output = init_field

        # 运行12小时模型
        for _ in range(num_12h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_12h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=12)
            # print("12", time)

        # 运行9小时模型
        for _ in range(num_9h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_9h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=9)
            # print("9", time)

        # 运行6小时模型
        for _ in range(num_6h):
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_6h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=6)
            # print("6", time)

        # 运行3小时模型
        if num_3h > 0:
            time_tag = torch.tensor([time.hour / 24, time.dayofweek / 7, time.dayofyear / 365.25]).to(device)
            output = model_3h(output.to(device), time_tag.unsqueeze(0))
            time = time + pd.Timedelta(hours=3)
            # print("3", time)

    results = output.detach().cpu().numpy().squeeze().copy()
    del output, init_field
    torch.cuda.empty_cache()
    return anti_stand(index, results)














