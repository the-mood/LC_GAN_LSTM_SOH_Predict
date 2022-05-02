"""
作者：杨文豪

描述：

时间：2022/4/13 20:34
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from data_process import battery_path
import os
import numpy as np
from scipy.signal import savgol_filter

plt.rcParams["font.family"] = "Kaiti"


# 画图--电池放电特征随循环次数的变化
def draw_fig_of_feature(feature_name):
    # 图的配置
    num = 0
    font_color = ['r-', 'g-', 'b-', 'y-']
    plt.title(feature_name + '随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel(feature_name)

    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        battery_data = pd.read_csv('./data/' + battery_name + '_discharge.csv')
        plt.plot(battery_data['discharge'], battery_data[feature_name], font_color[num], label=battery_name)
        plt.legend()
        num += 1
    plt.savefig("./image/feature/电池放电" + feature_name + "随循环变化图.jpg")
    plt.show()


# 画图--电池充电特征随循环次数的变化
def draw_fig_of_feature_charge(feature_name):
    # 图的配置
    num = 0
    font_color = ['r-', 'g-', 'b-', 'y-']
    plt.title(feature_name + '随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel(feature_name)
    # 第33次循环充电时间有异常，可以去掉
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        battery_data = pd.read_csv('./data/' + battery_name + '_charge.csv')
        plt.plot(battery_data['charge'], battery_data[feature_name], font_color[num], label=battery_name)
        plt.legend()
        num += 1
    plt.savefig("./image/feature/电池充电" + feature_name + "随循环次数变化图.jpg")
    plt.show()


# 可视化，某个电池在某一次放电时电压，电流，温度，电流负载，电压负载随时间的变化
def draw_fig_for_one_discharge(battery_name, discharge_num, feature_name):
    data = pd.DataFrame()
    dis = os.listdir('./data/discharge_data/' + battery_name)
    for i in dis:
        if 'dis_' + str(discharge_num) + '_' == i.split('cap')[0]:
            data = pd.read_csv('./data/discharge_data/' + battery_name + '/' + i)
            break
    # 图的配置
    plt.rcParams["font.family"] = "Kaiti"
    plt.title('第' + str(discharge_num) + '次循环中' + feature_name + '随时间的变化')
    plt.xlabel('时间')
    plt.ylabel(feature_name)
    plt.plot(data['time'], data[feature_name], 'r-')
    plt.savefig('./image/discharge_image/' + battery_name + '/' + feature_name + '/第' + str(discharge_num) + '次循环.jpg')
    plt.show()


# 可视化，某个电池在某一次充电时电压，电流，温度，电流负载，电压负载随时间的变化
def draw_fig_for_one_charge(battery_name, charge_num, feature_name):
    """特征名称随时间的变化曲线
    :param battery_name: 电池名称
    :param charge_num: 充电循环数
    :param feature_name: 特征名称
    :return:
    """
    if not os.path.exists('./image/charge_image/' + battery_name + '/' + feature_name):
        os.mkdir('./image/charge_image/' + battery_name + '/' + feature_name)
    data = pd.DataFrame()
    dis = os.listdir('./data/charge_data/' + battery_name)
    for i in dis:
        if str(charge_num) == i.split('.')[0].split('_')[-1]:
            data = pd.read_csv('./data/charge_data/' + battery_name + '/' + i)
            break
    # 图的配置
    plt.rcParams["font.family"] = "Kaiti"
    plt.title('第' + str(charge_num) + '次循环中' + feature_name + '随时间的变化')
    plt.xlabel('时间')
    plt.ylabel(feature_name)
    plt.plot(data['time'], data[feature_name], 'r-')
    plt.savefig('./image/charge_image/' + battery_name + '/' + feature_name + '/第' + str(charge_num) + '次循环.jpg')
    plt.show()


def draw_ic_curve(battery_name, discharge_num, feature_name):
    if not os.path.exists('./image/discharge_image/' + battery_name + '/' + feature_name):
        os.mkdir('./image/discharge_image/' + battery_name + '/' + feature_name)
    data = pd.DataFrame()
    dis = os.listdir('./data/discharge_data/' + battery_name)
    for i in dis:
        if 'dis_' + str(discharge_num) + '_' == i.split('cap')[0]:
            data = pd.read_csv('./data/discharge_data/' + battery_name + '/' + i)
            break
    # 图的配置
    plt.rcParams["font.family"] = "Kaiti"
    plt.title('第' + str(discharge_num) + '次循环中' + feature_name + '随时间电压的变化')
    plt.xlabel('电压')
    plt.xlim((2.65, 3.85))
    plt.ylabel(feature_name)
    voltage = list(data['voltage'])
    time = list(data['time'])
    end = list(voltage).index(np.min(voltage))
    t_time = np.diff(time[:end])
    t_vol = np.abs(np.diff(voltage[:end]))
    ic = 2 * np.divide(t_time, t_vol)
    ic = savgol_filter(ic[1:], 15, 3, mode='nearest')
    plt.plot(data['voltage'][1:len(ic)], ic[1:], 'r-')
    plt.savefig(
        './image/discharge_image/' + battery_name + '/' + feature_name + '/第' + str(discharge_num) + '次循环.jpg')
    plt.show()


# 画出第5，45，85，125，165次循环的ic曲线
def draw_ic_curve_compare():
    # 电池ic曲线对比
    b_names = ['B0005', 'B0006']
    nums = [5, 45, 85, 125, 165]
    # 图的配置
    color = ['r-', 'g-', 'b-', 'c-', 'y-']
    plt.rcParams["font.family"] = "Kaiti"
    for b_name in b_names:
        n = 0
        for num in nums:
            data = pd.DataFrame()
            dis = os.listdir('./data/discharge_data/' + b_name)
            for i in dis:
                if 'dis_' + str(num) + '_' == i.split('cap')[0]:
                    data = pd.read_csv('./data/discharge_data/' + b_name + '/' + i)
                    break
            plt.xlabel('电压')
            plt.xlim((2.65, 3.85))
            plt.ylabel('ic容量（dQ/dV）')
            plt.title('电池' + b_name + '的ic容量随电压的变化')
            voltage = list(data['voltage'])
            time = list(data['time'])
            end = list(voltage).index(np.min(voltage))
            t_time = np.diff(time[:end])
            t_vol = np.abs(np.diff(voltage[:end]))
            ic = 2 * np.divide(t_time, t_vol)
            ic = savgol_filter(ic[1:], 15, 3, mode='nearest')
            plt.plot(data['voltage'][1:len(ic)], ic[1:], color[n], label='第' + str(num) + '循环')
            plt.legend()
            n += 1
        plt.savefig('./image/feature/电池' + b_name + '的ic曲线对比')
        plt.show()


if __name__ == '__main__':
    # draw_ic_curve_compare()
    '''放电数据画图----开始'''
    # 画出所有电池的ic曲线
    # b_names = ['B0005', 'B0006']
    # for b_name in b_names:
    #     for i in range(1, 169):
    #         draw_ic_curve(b_name, i, 'ic')

    # fea = ['temperature', 'time', 'capacity', 'soh']
    # feature = ['voltage', 'temperature']
    # for name in fea:
    #     draw_fig_of_feature(name)

    # for b_name in b_names:
    #     for i in range(1, 169):
    #         for name in feature:
    #             draw_fig_for_one_discharge(b_name, i, name)

    # for name in feature:
    #     draw_fig_for_one_discharge(b_name, dis_num, name)
    '''放电数据画图----结束'''

    '''充电数据画图----开始'''
    # feature = ['voltage', 'current', 'temperature', 'current_charge', 'voltage_charge']
    # for b_name in b_names:
    #     for i in range(1, 171):
    #         for name in feature:
    #             draw_fig_for_one_charge(b_name, i, name)
    fea=['time','temperature']
    for f in fea:
        draw_fig_of_feature_charge(f)
    '''放电数据画图----结束'''
