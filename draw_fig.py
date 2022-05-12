"""
作者：杨文豪

描述：

时间：2022/4/13 20:34
"""
import math

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from data_process import battery_path
import os
import numpy as np
from scipy.signal import savgol_filter

b_names = ['B0005', 'B0006', 'B0007']
plt.rcParams["font.family"] = "Kaiti"
# 获取200种渐变色
clrs = []
for i in np.linspace(16711680, 255, 200):
    clrs.append('#%06x' % int(i))


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
    if not os.path.exists('./image/discharge_image/' + battery_name + '/' + feature_name):
        os.mkdir('./image/discharge_image/' + battery_name + '/' + feature_name)
    data = pd.DataFrame()
    dis = os.listdir('./data/discharge_data/' + battery_name)
    for i in dis:
        if 'dis_' + str(discharge_num) + '_' == i.split('cap')[0]:
            data = pd.read_csv('./data/discharge_data/' + battery_name + '/' + i)
            break
    # 图的配置
    plt.title('第' + str(discharge_num) + '次循环中' + feature_name + '随时间的变化')
    plt.xlabel('时间')
    plt.ylabel(feature_name)
    plt.plot(data['discharge_time'], data[feature_name], 'r-')
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
    plt.title('第' + str(charge_num) + '次循环中' + feature_name + '随时间的变化')
    plt.xlabel('时间')
    plt.ylabel(feature_name)
    plt.plot(data['charge_time'], data[feature_name], 'r-')
    plt.savefig('./image/charge_image/' + battery_name + '/' + feature_name + '/第' + str(charge_num) + '次循环.jpg')
    plt.show()


# 画出每次循环IC容量随时间的变化
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
    plt.title('第' + str(discharge_num) + '次循环中' + feature_name + '随时间电压的变化')
    plt.xlabel('电压')
    plt.xlim((2.65, 3.85))
    plt.ylabel(feature_name)
    voltage = list(data['voltage'])
    time = list(data['discharge_time'])
    end = list(voltage).index(np.min(voltage))
    t_time = np.diff(time[:end])
    t_vol = np.abs(np.diff(voltage[:end]))
    ic = 2 * np.divide(t_time, t_vol)
    ic = savgol_filter(ic[1:], 21, 3, mode='nearest')
    plt.plot(data['voltage'][1:len(ic)], ic[1:], 'r-')
    plt.savefig(
        './image/discharge_image/' + battery_name + '/' + feature_name + '/第' + str(discharge_num) + '次循环.jpg')
    plt.show()


# 除去异常值后查看ic容量的变化---电池放电IC容量随循环次数变化图.jpg
def draw_ic():
    b_names = ['B0005', 'B0006', 'B0007']
    color = ['r-', 'g-', 'b-', 'c-', 'y-']
    num = 0
    for battery_name in b_names:
        # data = pd.read_csv('./data/' + battery_name + '_discharge.csv')
        # data.at[60, 'ic'] = (data.at[61, 'ic'] + data.at[59, 'ic']) / 2

        data = pd.read_csv('./data/特征数据/' + battery_name + '_ic峰值.csv')
        data.at[60, 'ic_max'] = (data.at[61, 'ic_max'] + data.at[59, 'ic_max']) / 2

        plt.xlabel('循环次数')
        plt.ylabel('IC容量（dQ/dV）峰值')
        plt.title('电池' + battery_name + '的ic容量峰值随循环次数的变化')
        # data['ic']=savgol_filter(data['ic'],15,3,mode='nearest')
        # plt.plot(data['discharge'], data['ic'], color[num], label=battery_name)
        plt.plot(range(1, 169), data['ic_max'], color[num], label=battery_name)
        plt.legend()
        num += 1
    plt.savefig('./image/feature/电池放电IC容量随循环次数变化图.jpg')
    plt.show()


# 画出第5，45，85，125，165次循环的ic曲线
def draw_ic_curve_compare():
    # 电池ic曲线对比
    b_names = ['B0005', 'B0006', 'B0007']
    nums = range(1, 169)
    # nums = [5, 45, 85, 125, 165]
    # 图的配置
    # color = ['r-', 'g-', 'b-', 'c-', 'y-']
    for b_name in b_names:
        n = 0
        ic_max = []
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
            time = list(data['discharge_time'])
            end = list(voltage).index(np.min(voltage))
            t_time = np.diff(time[:end])
            t_vol = np.abs(np.diff(voltage[:end]))
            ic = 2 * np.divide(t_time, t_vol)
            ic = savgol_filter(ic[1:], 75, 3, mode='nearest')
            ic_max.append(np.max(ic))
            plt.plot(data['voltage'][1:len(ic)], ic[1:], clrs[n], label='第' + str(num) + '循环')
            plt.legend()
            n += 1
        # plt.savefig('./image/feature/电池' + b_name + '的ic曲线对比')
        ic_max = pd.DataFrame(ic_max, columns=['ic'])
        ic_max.to_csv('./data/特征数据/' + b_name + '_ic峰值.csv', index=False, header=['ic_max'])
        plt.show()


def draw_CC_ratio():
    plt.title('电池恒流充电时间占总充电时间的比值随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel('恒流充电时间占总充电时间的比值')
    n = 0
    for battery_name in b_names:
        data = pd.read_csv('./data/特征数据/' + battery_name + '_CC_ratio.csv')
        plt.plot(range(1, 168), data['CC_ratio'], clrs[n], label=battery_name)
        plt.legend()
        n += 45
    plt.savefig('./image/feature/电池恒流充电时间占总充电时间的比值随循环次数的变化.jpg')
    plt.show()


# 电压到达4.2v的时间随循环次数的变化
def draw_voltage_compare():
    plt.xlabel('循环次数')
    plt.ylabel('电压到达4.2v的时间')
    n = 0
    for b_name in ['B0005', 'B0006', 'B0007']:
        plt.title('电池' + b_name + '电压到达4.2v的时间随循环次数的变化')
        file_names = os.listdir('./data/charge_data/' + b_name)
        file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        t = []
        for file_name in file_names:
            if file_name != 'charge_33.csv' and file_name != 'charge_170.csv' and file_name != 'charge_1.csv':
                data = pd.read_csv('./data/charge_data/' + b_name + '/' + file_name)
                data = data.round(decimals=2)
                index = list(data['voltage']).index(float(4.2))
                # print(index)
                t.append(data.at[index, 'charge_time'])
        # 将电池充电到4.2v的时间分别画在一张图上
        plt.plot(range(1, 168), t, clrs[n])
        plt.savefig('./image/feature/电池' + b_name + '电压到达4.2v的时间随循环次数的变化.jpg')
        # temp = pd.DataFrame(t, columns=['charge_to_4.2v_time'])
        # temp.to_csv('./data/特征数据/' + b_name + '_charge_to_4.2v_time.csv', index=False, header=['charge_to_4.2v_time'])

        # 将三个电池充电到4.2v的时间画在一张图上，查看趋势
        # plt.plot(range(1, 168), t, clrs[n], label=b_name)
        # plt.legend()
        # n += 1
        # voltage=data[]

        # 选择几个循环观看充电到4.2v的时间对比
        # num = 0
        # for file_name in file_names:
        #     if int(file_name.split('.')[0].split('_')[-1]) % 20 == 5:
        #         data = pd.read_csv('./data/charge_data/B0005/' + file_name)
        #         plt.plot(data['charge_time'], data['voltage'], color=clrs[num], label=file_name)
        #         plt.legend()
        #     num += 1
    # 将三个电池充电到4.2v的时间画在一张图上，查看趋势
    # plt.savefig('./image/feature/电池充电时电压达到4.2v的时间随循环次数的变化.jpg')
    # plt.show()


# 放电电压达到最小值所用的时间随循环次数的变化
def draw_discharge_voltage_compare():
    # plt.title('放电电压达到最小值所用的时间随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel('放电电压达到最小值所用的时间')
    n = 0
    for b_name in ['B0005', 'B0006', 'B0007']:
        plt.title('电池' + b_name + '放电电压达到最小值所用的时间随循环次数的变化')
        file_names = os.listdir('./data/discharge_data/' + b_name)
        file_names.sort(key=lambda x: int(x.split('_')[1]))
        # 三个电池画在三张图上
        t = []
        for file_name in file_names:
            data = pd.read_csv('./data/discharge_data/' + b_name + '/' + file_name)
            # 电池放电时电压达到最小值时所用的时间
            time = data.at[data['voltage'].idxmin(), 'discharge_time']
            t.append(time)
        plt.plot(range(1, 169), t, clrs[n])
        n += 1
        plt.savefig('./image/feature/电池' + b_name + '放电电压达到最小值所用的时间随循环次数的变化')
        temp = pd.DataFrame(t, columns=['discharge_to_min_voltage_time'])
        temp.to_csv('./data/特征数据/' + b_name + '_discharge_to_min_voltage_time.csv', index=False,
                    header=['discharge_to_min_voltage_time'])

        # 从电池循环种选择几个观看趋势
        # num = 0
        # for file_name in file_names:
        #     if num%10==0:
        #         data = pd.read_csv('./data/discharge_data/B0005/' + file_name)
        #         plt.plot(data['discharge_time'], data['voltage'], color=clrs[num], label=file_name.split('_')[1])
        #         plt.legend()
        #     num += 1
        plt.show()


# 恒流充电用时随循环次数的变化
def draw_current_compare():
    # plt.title('充电时恒流充电的时间随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel('恒流充电时间')
    for b_name in b_names:
        plt.title('电池' + b_name + '充电时恒流充电的时间随循环次数的变化')
        file_names = os.listdir('./data/charge_data/' + b_name)
        file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        t = []
        time = []
        ratio = []
        for file_name in file_names:
            # and file_name != 'charge_1.csv'
            if file_name != 'charge_33.csv' and file_name != 'charge_170.csv' and file_name != 'charge_1.csv':
                data = pd.read_csv('./data/charge_data/' + b_name + '/' + file_name)
                data = data.round(decimals=2)
                # 记录索引值
                num = 0
                for cur in data['current_charge'][2:]:
                    if 1.50 <= cur <= 1.51:
                        num += 1
                # t恒流充电时间
                t.append(data.at[num, 'charge_time'])
                # time总共的充电时间
                time.append(data['charge_time'].max())
                ratio.append(round(data['charge_time'][num] / data['charge_time'].max(), 2))
        plt.plot(range(1, 168), t, 'r-')
        plt.savefig('./image/feature/电池' + b_name + '恒流充电时间随循环次数的变化.jpg')
        plt.show()
        t = pd.DataFrame(t, columns=['CC_time'])
        time = pd.DataFrame(time, columns=['all_charge_time'])
        ratio = pd.DataFrame(ratio, columns=['CC_ratio'])
        temp = t.join(time, how='left').join(ratio, how='left')
        temp[['CC_time', 'all_charge_time', 'CC_ratio']] \
            .to_csv('./data/特征数据/' + b_name + '_CC_ratio.csv', index=False,
                    header=['CC_time', 'all_charge_time', 'CC_ratio'])

    # num = 0
    # for file_name in file_names:
    #     if num % 20 == 5:
    #         data = pd.read_csv('./data/charge_data/B0005/' + file_name)
    #         plt.plot(data['charge_time'], data['current'], color=clrs[num], label=file_name.split('.')[0])
    #         plt.legend()
    #     num += 1


def draw_temperature_compare():
    plt.title('放电温度随时间的变化')
    plt.xlabel('时间')
    plt.ylabel('温度')
    file_names = os.listdir('./data/discharge_data/B0005')
    # file_names.remove('.DS_Store')
    file_names.sort(key=lambda x: int(x.split('_')[1]))
    num = 0
    for file_name in file_names:
        if num % 20 == 5:
            data = pd.read_csv('./data/discharge_data/B0005/' + file_name)
            # temperature=np.mean(list(data['temperature']))
            # t.append(temperature)
            plt.plot(data['discharge_time'], data['temperature'], color=clrs[num], label=file_name.split('.')[0])
            plt.legend()
        num += 1

    plt.show()


def draw_charge_temperature_compare():
    plt.title('充电温度随时间的变化')
    plt.xlabel('时间')
    plt.ylabel('温度')
    file_names = os.listdir('./data/charge_data/B0005')
    # file_names.remove('.DS_Store')
    file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    num = 0
    t = []
    for file_name in file_names:
        # if num % 20 == 5:
        data = pd.read_csv('./data/charge_data/B0005/' + file_name)
        temperature = np.mean(list(data['temperature']))
        t.append(temperature)
    plt.plot(range(1, 171), t, 'r-')
    # plt.plot(data['charge_time'], data['temperature'], color=clrs[num], label=file_name.split('.')[0])
    # plt.legend()
    # num += 1

    plt.show()


if __name__ == '__main__':
    # draw_discharge_voltage_compare()
    # draw_voltage_compare()
    draw_current_compare()

    # draw_charge_temperature_compare()
    # draw_temperature_compare()
    # draw_ic_curve_compare()

    # draw_CC_ratio()
    '''放电数据画图----开始'''
    # draw_ic()

    # 画出所有电池的ic曲线
    # b_names = ['B0005', 'B0006', 'B0007']
    # for b_name in b_names:
    #     for i in range(1, 169):
    #         draw_ic_curve(b_name, i, 'ic')

    # fea = ['temperature', 'discharge_time', 'capacity', 'soh', 'ic']
    # for name in fea:
    #     draw_fig_of_feature(name)

    # b_names = ['B0007']
    # feature = ['voltage', 'temperature']
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
    # fea=['time','temperature']
    # for f in fea:
    #     draw_fig_of_feature_charge(f)
    '''放电数据画图----结束'''
