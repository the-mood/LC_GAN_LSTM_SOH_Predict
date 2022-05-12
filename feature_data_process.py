"""
作者：杨文豪

描述：对提取出的特征数据进行处理

时间：2022/5/10 14:34
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

b_names = ['B0005', 'B0006', 'B0007']
feature = ['discharge', 'charge_to_4.2v_time', 'CC_ratio', 'discharge_to_min_voltage_time',
           'ic_max', 'discharge_time', 'capacity', 'soh']
plt.rcParams["font.family"] = "Kaiti"


# 向电池充电到4.2v的时间的特征数据中插入第33次循环的数据，值取32，34次循环的均值
def process_charge_voltage():
    for b_name in b_names:
        data = pd.read_csv('./data/特征数据/' + b_name + '_charge_to_4.2v_time.csv')
        # 插入数据
        # value = round((data['charge_to_4.2v_time'][30] + data['charge_to_4.2v_time'][31]) / 2, 2)
        # obj = {'charge_to_4.2v_time': value}
        # data = data[:31].append(obj, ignore_index=True).append(data[31:], ignore_index=True)

        # 修改数据
        data['charge_to_4.2v_time'][11] = (data['charge_to_4.2v_time'][10] + data['charge_to_4.2v_time'][12]) / 2
        data.to_csv('./data/特征数据/' + b_name + '_charge_to_4.2v_time.csv', index=False, header=['charge_to_4.2v_time'])


# 向电池恒流充电时间的特征数据中插入第33次循环的数据，值取32，34次循环的均值
def process_CC_ratio():
    for b_name in b_names:
        data = pd.read_csv('./data/特征数据/' + b_name + '_CC_ratio.csv')
        CC_value = round((data['CC_time'][30] + data['CC_time'][31]) / 2, 2)
        all_charge_value = round((data['all_charge_time'][30] + data['all_charge_time'][31]) / 2, 2)
        ratio_value = round(CC_value / all_charge_value, 2)
        obj = {'CC_time': CC_value, 'all_charge_time': all_charge_value, 'CC_ratio': ratio_value}
        data = data[:31].append(obj, ignore_index=True).append(data[31:], ignore_index=True)
        data.to_csv('./data/特征数据/' + b_name + '_CC_ratio.csv',
                    index=False, header=['CC_time', 'all_charge_time', 'CC_ratio'])


# 处理ic_max中的异常值
def process_ic():
    for b_name in b_names:
        data = pd.read_csv('./data/特征数据/' + b_name + '_ic峰值.csv')
        data['ic_max'][60] = (data['ic_max'][59] + data['ic_max'][61]) / 2
        data = data.round(2)
        data.to_csv('./data/特征数据/' + b_name + '_ic峰值.csv', index=False, header=['ic_max'])


def process_ic_max():
    for b_name in b_names:
        # data = pd.read_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')
        data = pd.read_csv('./data/特征数据/' + b_name + '_ic峰值.csv')
        ic_max = data['ic_max']
        ic_max = savgol_filter(ic_max, 21, 1, mode='nearest')
        plt.title('滤波前后对比')
        plt.xlabel('循环次数')
        plt.ylabel('ic_max')
        plt.plot(range(1, 169), data['ic_max'], 'g-', label='真实数据')
        plt.legend()
        plt.plot(range(1, 169), ic_max, 'r-', label='滤波后的数据')
        plt.legend()
        plt.show()
        # data['ic_max'] = ic_max
        # data.to_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')


def process_cc_ratio():
    for b_name in b_names:
        data = pd.read_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')
        # data = pd.read_csv('./data/特征数据/' + b_name + '_ic峰值.csv')
        CC_ratio = data['CC_ratio']
        CC_ratio = savgol_filter(CC_ratio, 15, 1, mode='nearest')
        plt.title('滤波前后对比')
        plt.xlabel('循环次数')
        plt.ylabel('ic_max')
        plt.plot(range(1, 169), data['CC_ratio'], 'g-', label='真实数据')
        plt.legend()
        plt.plot(range(1, 169), CC_ratio, 'r-', label='滤波后的数据')
        plt.legend()
        plt.show()
        # data['CC_ratio'] = CC_ratio
        # data.to_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')


# 将所有特征数据拼接起来
def concact_all_data():
    for b_name in b_names:
        data_discharge = pd.read_csv('./data/' + b_name + '_discharge.csv')
        discharge = pd.DataFrame(data_discharge['discharge'], columns=['discharge'])
        data_charge_vol = pd.read_csv('./data/特征数据/' + b_name + '_charge_to_4.2v_time.csv')
        data_CC = pd.read_csv('./data/特征数据/' + b_name + '_CC_ratio.csv')
        data_dis_min = pd.read_csv('./data/特征数据/' + b_name + '_discharge_to_min_voltage_time.csv')
        data_ic = pd.read_csv('./data/特征数据/' + b_name + '_ic峰值.csv')
        discharge_time = pd.DataFrame(data_discharge['discharge_time'], columns=['discharge_time'])
        capacity = pd.DataFrame(data_discharge['capacity'], columns=['capacity'])
        soh = pd.DataFrame(data_discharge['soh'], columns=['soh'])

        temp = discharge.join(data_charge_vol, how='left').join(data_CC, how='left') \
            .join(data_dis_min, how='left').join(data_ic, how='left').join(discharge_time, how='left') \
            .join(capacity, how='left').join(soh, how='left')
        temp[feature].to_csv('./data/all_feature_data/' + b_name + '_all_feature.csv', index=False, header=feature)


# 使用滑动窗口扩展数据
def slide_window_to_extend_data():
    num = 5
    for b_name in b_names:
        b_data = pd.read_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')
        temp_data = b_data[0:117]
        for i in range(1, 51):
            temp_data = pd.concat([temp_data, b_data[i:i + 117]], axis=0)

        temp_data[feature] \
            .to_csv('./data/extend_data/b_00' + str(num) + '_extend_all_feature.csv',
                    index=False, header=feature)
        num += 1


if __name__ == '__main__':
    # process_charge_voltage()
    # process_CC_ratio()
    # concact_all_data()
    # slide_window_to_extend_data()
    # process_ic_max()
    # process_ic()
    process_cc_ratio()
