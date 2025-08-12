import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt
from utils.init import get_vars
from utils.handle import open_rean_merra, open_clim_merra

warnings.filterwarnings("ignore")


def rmse(y, y_hat, region = None):

    """
    计算给定y与y_hat之间的加权RMSE
    返回一个元素为channels，长度为time的列表
    """
    if region:
        lat = np.arange(region[2], region[3]+0.5, 0.5)
        lat = lat[::-1]
    else:
        lat = np.arange(-90, 90.5, 0.5)
    N_lat = len(lat)
    # 计算每个纬度的权重
    cos_lat = np.cos(np.deg2rad(lat))
    sum_cos_lat = np.sum(cos_lat)
    weights = N_lat * (cos_lat / sum_cos_lat)

    # 初始化一个长度为time的列表来存储结果
    time = len(y)
    rmse_list = []

    # 对每个时间步进行计算
    for t in range(time):
        m = y_hat[t]
        n = y[t]
        if (y[t] is not None) and (y_hat[t] is not None):

            # 取出当前时间步的真实值和预测值
            y_t = np.array(y[t])  # [channels, height, width]
            y_hat_t = np.array(y_hat[t])  # [channels, height, width]

            # 计算误差
            error = y_t - y_hat_t

            # 将误差平方后乘以纬度权重 (weights 在第二个维度（height）上应用)
            weighted_error_square = error ** 2 * weights[:, np.newaxis]

            # 对每个通道分别计算加权的MSE (在 height 和 width 维度上)
            mse_per_channel = np.mean(weighted_error_square, axis=(1, 2))

            # 对每个通道取平方根，得到加权RMSE
            rmse_per_channel = np.sqrt(mse_per_channel)

            # 将结果从 [channels] 变成 [channels, 1] 的大小
            rmse_per_channel = rmse_per_channel.reshape(-1, 1)

            # 将当前时间步的 RMSE 添加到列表中
            rmse_list.append(rmse_per_channel)

        else:
            rmse_list.append(None)

    return rmse_list


def correlation(y, y_hat):
    """
    计算给定y与y_hat之间的Pearson相关性系数
    返回一个元素为channels，长度为time的列表
    """

    # 初始化一个长度为time的列表来存储结果
    time = len(y)
    correlation_list = []

    # 对每个时间步进行计算
    for t in range(time):

        if (y[t] is not None) and (y_hat[t] is not None):

            # 取出当前时间步的真实值和预测值
            y_t = np.array(y[t])  # [channels, height, width]
            y_hat_t = np.array(y_hat[t])  # [channels, height, width]

            # 初始化一个存储每个通道相关系数的列表
            corr_per_channel = []

            # 对每个通道分别计算相关性
            for c in range(y_t.shape[0]):  # 遍历 channels
                y_t_flat = y_t[c].flatten()  # 展平为1D数组
                y_hat_t_flat = y_hat_t[c].flatten()  # 展平为1D数组

                # 计算相关系数，使用 np.corrcoef 计算相关性矩阵并取出相关系数
                corr_matrix = np.corrcoef(y_t_flat, y_hat_t_flat)
                corr = corr_matrix[0, 1]  # 取出相关系数

                corr_per_channel.append(corr)

            # 将结果从 [channels] 变成 [channels, 1] 的大小
            corr_per_channel = np.array(corr_per_channel).reshape(-1, 1)

            # 将当前时间步的相关系数添加到列表中
            correlation_list.append(corr_per_channel)

        else:
            correlation_list.append(None)

    return correlation_list



def get_mean_state(region = None):

    """
    计算所有特征的平均态，与样本结构保持一直
    """

    # 时间
    times = ["01:30", "04:30", "07:30", "10:30", "13:30", "16:30", "19:30", "22:30"]
    times = pd.to_datetime([f"2000-01-01 {time}" for time in times])

    # 变量
    var_dict = get_vars()

    mean_matrix = []

    for time in times:
        matrix, index, _, _ = open_clim_merra(time.strftime('%Y-%m-%d %H:%M'), region=region)
        mean_matrix.append(matrix)
    mean_matrix = np.stack(mean_matrix, axis=0)  # [time, channels, height, width]

    return mean_matrix, index



def acc(y, y_hat, index, index_loc, mean_state, ground_truth_time, region = None):
    """
    计算每个时间点和每个特征的Anomaly Correlation Coefficient (ACC)，并考虑纬度权重。

    参数：
    y: 实况观测值，长度为time的列表，每个元素是[channels, height, width]的矩阵
    y_hat: 预测值，长度为time的列表，每个元素是[channels, height, width]的矩阵
    index: 特征名字，对应channels
    index_loc: 在mean_state中索引特征位置的索引列表
    mean_state: 气候平均态，大小为[time, channels, height, width]的矩阵
    ground_truth_time: 长度为time的时间字符串列表

    返回：
    acc_list: 长度为time的列表，每个元素是(channels, 1)大小维度的矩阵，表示该时间点的各个特征的ACC
    """

    # 定义纬度权重
    if region:
        lat = np.arange(region[2], region[3] + 0.5, 0.5)
    else:
        lat = np.arange(-90, 90.5, 0.5)
    N_lat = len(lat)
    # 计算每个纬度的权重 (是一个反二次曲线)
    cos_lat = np.cos(np.deg2rad(lat))
    sum_cos_lat = np.sum(cos_lat)
    weights = N_lat * (cos_lat / sum_cos_lat)

    # 初始化一个空的列表来保存所有时间点的ACC结果
    acc_list = []

    for t in range(len(y)):

        if (y[t] is not None) and (y_hat[t] is not None):

            obs = y[t]  # 实况观测，维度为 [channels, height, width]
            pred = y_hat[t]  # 预测值，维度为 [channels, height, width]

            # 获取对应时间的气候平均态，维度为 [channels, height, width]
            time_str = ground_truth_time[t][-4:]  # 提取时间字符串，形如 '0130', '0430' 等
            time_index = ["0130", "0430", "0730", "1030", "1330", "1630", "1930", "2230"].index(time_str)
            mean = mean_state[time_index]
            mean = mean[index_loc]

            # 计算观测值和预测值的异常 (去掉气候平均态)
            obs_anomaly = obs - mean  # [channels, height, width]
            pred_anomaly = pred - mean  # [channels, height, width]

            # 初始化ACC的矩阵，维度为 [channels, 1]
            acc_matrix = np.zeros((obs.shape[0], 1))

            # 对每个特征 (channel) 计算加权ACC
            for c in range(obs.shape[0]):  # 遍历每个特征
                obs_anomaly_weighted = obs_anomaly[c] * weights[:, np.newaxis]
                pred_anomaly_weighted = pred_anomaly[c] * weights[:, np.newaxis]

                obs_anomaly_flat = obs_anomaly_weighted.flatten()  # 展平成一维数组
                pred_anomaly_flat = pred_anomaly_weighted.flatten()  # 展平成一维数组

                # 计算ACC的分子和分母，并进行加权处理
                numerator = np.sum(obs_anomaly_flat * pred_anomaly_flat)
                denominator = np.sqrt(np.sum(obs_anomaly_flat ** 2) * np.sum(pred_anomaly_flat ** 2))

                # 防止除零错误
                if denominator != 0:
                    acc_matrix[c] = numerator / denominator
                else:
                    acc_matrix[c] = 0  # 如果分母为0，ACC设置为0

            # 将当前时间点的ACC矩阵加入列表
            acc_list.append(acc_matrix)

        else:
            acc_list.append(None)

    return acc_list



def training_log(log_path):

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
    plt.savefig("/data/_project_zxt/merra_forecast/log/log.png", dpi=500)








if __name__ == "__main__":

    # mean_state_cal(1980, 2021)
    # acc(None, None, None)
    # get_mean_state()
    # rmse(None, None)
    training_log("/data/_project_zxt/merra_forecast/log/log_12h.txt")