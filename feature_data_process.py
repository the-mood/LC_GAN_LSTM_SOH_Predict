"""
作者：杨文豪

描述：对提取出的特征数据进行处理

时间：2022/5/10 14:34
"""
import os
import numpy as np
import pandas as pd

b_names = ['B0005', 'B0006', 'B0007']
feature = ['discharge', 'charge_to_4.2v_time', 'CC_ratio', 'discharge_to_min_voltage_time',
           'ic_max', 'discharge_time', 'capacity', 'soh']


# 向电池充电到4.2v的时间的特征数据中插入第33次循环的数据，值取32，34次循环的均值
def process_charge_voltage():
    for b_name in b_names:
        data = pd.read_csv('./data/特征数据/' + b_name + '_charge_to_4.2v_time.csv')
        value = round((data['charge_to_4.2v_time'][30] + data['charge_to_4.2v_time'][31]) / 2, 2)
        obj = {'charge_to_4.2v_time': value}
        data = data[:31].append(obj, ignore_index=True).append(data[31:], ignore_index=True)
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
        b_data = pd.read_csv('./data/all_feature_data/'+b_name+'_all_feature.csv')
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
    slide_window_to_extend_data()