"""
作者：杨文豪

描述：用来处理NASA电池数据集和牛津电池数据集
    电池数据为.mat文件
时间：2022/3/31 9:27
"""
import glob
import numpy as np
import scipy.io as sio  # 读取mat文件
import os
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter

base_path = r'D:\当前可能用到的文件\电池数据集\NASA_data'
f_name = 'BatteryAgingARC-FY08Q4'
# 获取当前路径下所有的.mat后缀的文件
battery_path = glob.glob(os.path.join(base_path + '\\' + f_name, '*.mat'))

'''对电池放电数据的处理---开始'''

fea = ['discharge', 'voltage', 'temperature', 'time', 'capacity']
# b05,b06,b07
# 反标准化所需
max_dis = 168
min_dis = 1
max_vol = [3.621191033, 3.697169686, 3.428899305]
min_vol = [3.205068916, 2.120698095, 1.813269431]
max_tem = [41.45023192, 42.00754045, 42.33252237]
min_tem = [37.80133585, 36.98024266, 38.34572084]
max_time = 3690.234
min_time = 2792.485
max_cap = [1.856487421, 2.035337591, 1.891052295]
min_cap = [1.287452522, 1.153818332, 1.40045524]


# 获取电池放电数据
def get_data():
    '''
    将电池数据以字典类型存储
         battery_data=[dict0,dict1,dict2,dict3]
        dict0={'B0005': [{'discharge': 1, 'capacity': [1.8564874208181574]},
                {'discharge': 2, 'capacity': [1.846327249719927]},....]
    :return:
    '''
    # 存放base_path下所有电池的循环次数和容量数据
    battery_data = []
    # 获取电池的具体名称
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        temp_data = {battery_name: process_data(f_name, battery_name)}
        battery_data.append(temp_data)
    return battery_data


# 对电池的数据进行处理
def process_data(file_name, b_name):
    '''
    :param file_name: 电池数据所在的文件名
    :param b_name: 电池名称
        将每次放电的数据存入csv中
            数据包括：电压，电流，温度，电流负载，电压负载，时间
    :return:
    '''
    if not os.path.exists('./data/discharge_data/' + b_name):
        os.mkdir('./data/discharge_data/' + b_name)
    # 获取电池数据的地址
    data_path = base_path + '\\' + file_name + '\\' + b_name + '.mat'
    # 使用matlab加载数据集==》字典类型
    data = sio.matlab.loadmat(data_path)
    # 从获取的字典中获取b_name电池数据
    data_B = data.get(b_name)
    battery = []
    b_data = []
    discharge_Num = 0
    # 获取放电循环中的时间和容量
    for i in range(0, len(data_B[0][0][0][0])):
        if data_B[0][0][0][0][i][0][0] == 'discharge':
            discharge_Num += 1
            temp_dict = {'discharge': discharge_Num,
                         'voltage': np.array(data_B[0][0][0][0][i][3][0][0][0][0]),
                         'current': np.array(data_B[0][0][0][0][i][3][0][0][1][0]),
                         'temperature': np.array(data_B[0][0][0][0][i][3][0][0][2][0]),
                         'current_load': np.array(data_B[0][0][0][0][i][3][0][0][3][0]),
                         'voltage_load': np.array(data_B[0][0][0][0][i][3][0][0][4][0]),
                         'time': np.array(data_B[0][0][0][0][i][3][0][0][5][0]),
                         'capacity': list(data_B[0][0][0][0][i][3][0][0][6][0])
                         }
            battery.append(temp_dict)
            # 将每一次放电循环的数据存入csv
            data_to_csv(temp_dict, b_name)

            # 处理数据获取ic容量曲线峰值
            time = temp_dict.get('time')
            voltage = temp_dict.get('voltage')
            end = list(voltage).index(np.min(voltage))
            t_time = np.diff(time[:end])
            t_vol = np.abs(np.diff(voltage[:end]))
            ic = 2 * np.divide(t_time, t_vol)
            # ic = savgol_filter(ic, 15, 3, mode='nearest')
            ic = np.max(ic[1:])
            t_dict = {'discharge': discharge_Num,
                      'temperature': np.max(np.array(data_B[0][0][0][0][i][3][0][0][2][0])),
                      'time': np.array(data_B[0][0][0][0][i][3][0][0][5][0])[-1],
                      'capacity': list(data_B[0][0][0][0][i][3][0][0][6][0]),
                      'ic': ic
                      }
            b_data.append(t_dict)
    return b_data


# 将获取的数据存入到csv文件中
def data_to_csv(data, battery_name):
    '''
    :param data: 某一次放电的数据
    :param battery_name: 电池名称
    :return:
    '''
    temp_vol = data.get('voltage')
    temp_cur = data.get('current')
    temp_tem = data.get('temperature')
    temp_cur_load = data.get('current_load')
    temp_vol_load = data.get('voltage_load')
    temp_time = data.get('time')
    temp_ic = np.abs(2 * np.divide(np.diff(temp_time), np.diff(temp_vol)))
    temp_vol = pd.DataFrame(temp_vol, columns=['voltage'])
    temp_cur = pd.DataFrame(temp_cur, columns=['current'])
    temp_tem = pd.DataFrame(temp_tem, columns=['temperature'])
    temp_cur_load = pd.DataFrame(temp_cur_load, columns=['current_load'])
    temp_vol_load = pd.DataFrame(temp_vol_load, columns=['voltage_load'])
    temp_time = pd.DataFrame(temp_time, columns=['time'])
    temp_ic = pd.DataFrame(temp_ic, columns=['ic'])
    temp = temp_vol.join(temp_cur, how='left').join(temp_tem, how='left').join(temp_cur_load, how='left') \
        .join(temp_vol_load, how='left').join(temp_time, how='left').join(temp_ic, how='left')
    temp[['voltage', 'current', 'temperature', 'current_load', 'voltage_load', 'time', 'ic']] \
        .to_csv(
        './data/discharge_data/' + battery_name + '/dis_' + str(data.get('discharge')) + '_cap_' + str(
            data.get('capacity')[0]) + '.csv',
        index=False,
        header=['voltage', 'current', 'temperature', 'current_load', 'voltage_load', 'time', 'ic'])


# 将放电数据--特征数据存入csv中
def discharge_battery_data_to_csv(data):
    '''
    :param data:    data=get_data()
    :return: 将放电数据--特征数据存入csv中
    '''
    num = 0
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        dis = []
        vol = []
        tem = []
        time = []
        cap = []
        ic = []
        soh = []
        for i in range(0, len(data[num].get(battery_name))):
            dis.append(data[num].get(battery_name)[i].get('discharge'))
            tem.append(data[num].get(battery_name)[i].get('temperature'))
            time.append(data[num].get(battery_name)[i].get('time'))
            cap.append(data[num].get(battery_name)[i].get('capacity')[0])
            ic.append(data[num].get(battery_name)[i].get('ic'))
        for j in range(0, len(cap)):
            soh.append((cap[j] / cap[0]) * 100)
        num += 1
        dis = pd.DataFrame(dis, columns=['discharge'])
        tem = pd.DataFrame(tem, columns=['temperature'])
        time = pd.DataFrame(time, columns=['time'])
        cap = pd.DataFrame(cap, columns=['capacity'])
        soh = pd.DataFrame(soh, columns=['soh'])
        ic = pd.DataFrame(ic, columns=['ic'])
        temp = dis.join(vol, how='left').join(tem, how='left').join(time, how='left') \
            .join(cap, how='left').join(soh, how='left').join(ic, how='left')
        temp[['discharge', 'temperature', 'time', 'capacity', 'soh', 'ic']] \
            .to_csv('./data/' + battery_name + '_discharge.csv', index=False,
                    header=['discharge', 'temperature', 'time', 'capacity', 'soh', 'ic'])


# 将数据标准化以后存入csv
def Standard_data_to_csv(data):
    num = 7
    for b_data in data:
        for name in fea:
            min = float(b_data[name].min())
            max = float(b_data[name].max())
            b_data[name] = b_data[name].astype('float')
            for i in range(0, 168):
                b_data[name][i] = (b_data[name][i] - min) / (max - min)
        b_data[fea].to_csv('./data/Standard_data/b0' + str(num) + '.csv', index=False,
                           header=['discharge', 'voltage', 'temperature', 'time', 'capacity'])
        num -= 1


# 使用滑动窗口扩展数据
def slide_window_to_extend_data(data):
    '''
    :param data: pd.DataFrame格式，discharge、capacity、soh
        将扩展以后的数据写入csv
    :return:
    '''
    num = 5
    for b_data in data:
        temp_data = b_data[0:117]
        for i in range(1, 51):
            temp_data = pd.concat([temp_data, b_data[i:i + 117]], axis=0)

        temp_data[['discharge', 'voltage', 'temperature', 'time', 'capacity', 'soh']] \
            .to_csv('./data/extend_data/b_0' + str(num) + '.csv',
                    index=False, header=['discharge', 'voltage', 'temperature', 'time', 'capacity', 'soh'])
        num += 1


# 使用滑动窗口对IC进行扩展
def slide_window_to_extend_IC_data():
    data_5 = pd.read_csv('./data/B0005_discharge.csv')
    data_6 = pd.read_csv('./data/B0006_discharge.csv')
    data_5 = data_5.drop(data_5.index[data_5['discharge'] == 61])
    data_6 = data_6.drop(data_6.index[data_6['discharge'] == 61])
    data = [data_5, data_6]
    num = 5
    for b_data in data:
        temp_data = b_data[0:117]
        for i in range(1, 51):
            temp_data = pd.concat([temp_data, b_data[i:i + 117]], axis=0)
        temp_data[['discharge', 'ic']] \
            .to_csv('./data/extend_data/b_0' + str(num) + '_dis_ic.csv',
                    index=False, header=['discharge', 'ic'])
        num += 1


'''对电池放电数据的处理---结束'''

'''对电池充电数据的处理---开始'''


# 将获取的电池充电数据以字典形式返回
def get_charge_data():
    """
    将电池充电数据以字典类型存储
         battery_data=[dict0,dict1,dict2,dict3]
        dict0={'B0005': [{'charge': 1, 'charge': [1.8564874208181574]}, 'time':
                {'charge': 2, 'charge': [1.846327249719927]},....]
    :return:
    """
    # 存放base_path下所有电池的循环次数和容量数据
    battery_data = []
    # 获取电池的具体名称
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        temp_data = {battery_name: process_charge_data(f_name, battery_name)}
        battery_data.append(temp_data)
    return battery_data


# 对充电数据进行处理--获取充电数据中可用的特征数据
def process_charge_data(file_name, b_name):
    """
        :param file_name: 电池数据所在的文件名
        :param b_name: 电池名称
            将每次放电的数据存入csv中
                数据包括：电压，电流，温度，电流负载，电压负载，时间
        :return:
        """
    if not os.path.exists('./data/charge_data/' + b_name):
        os.mkdir('./data/charge_data/' + b_name)
    # 获取电池数据的地址
    data_path = base_path + '\\' + file_name + '\\' + b_name + '.mat'
    # 使用matlab加载数据集==》字典类型
    data = sio.matlab.loadmat(data_path)
    # 从获取的字典中获取b_name电池数据
    data_B = data.get(b_name)
    battery = []
    b_data = []
    charge_Num = 0
    # 获取放电循环中的时间和容量
    for i in range(0, len(data_B[0][0][0][0])):
        if data_B[0][0][0][0][i][0][0] == 'charge':
            charge_Num += 1
            temp_dict = {'charge': charge_Num,
                         'voltage': np.array(data_B[0][0][0][0][i][3][0][0][0][0]),
                         'current': np.array(data_B[0][0][0][0][i][3][0][0][1][0]),
                         'temperature': np.array(data_B[0][0][0][0][i][3][0][0][2][0]),
                         'current_charge': np.array(data_B[0][0][0][0][i][3][0][0][3][0]),
                         'voltage_charge': np.array(data_B[0][0][0][0][i][3][0][0][4][0]),
                         'time': np.array(data_B[0][0][0][0][i][3][0][0][5][0])
                         }
            battery.append(temp_dict)
            charge_data_to_csv(temp_dict, b_name)

            t_dict = {'charge': charge_Num,
                      'temperature': np.max(np.array(data_B[0][0][0][0][i][3][0][0][2][0])),
                      'time': np.array(data_B[0][0][0][0][i][3][0][0][5][0])[-1],
                      }
            b_data.append(t_dict)
    return b_data


# 将获取的数据存入到csv文件中
def charge_data_to_csv(data, battery_name):
    """
    :param data: 某一次充电循环的数据
    :param battery_name: 电池名称
    :return:
    """
    temp_vol = data.get('voltage')
    temp_cur = data.get('current')
    temp_tem = data.get('temperature')
    temp_cur_load = data.get('current_charge')
    temp_vol_load = data.get('voltage_charge')
    temp_time = data.get('time')
    temp_vol = pd.DataFrame(temp_vol, columns=['voltage'])
    temp_cur = pd.DataFrame(temp_cur, columns=['current'])
    temp_tem = pd.DataFrame(temp_tem, columns=['temperature'])
    temp_cur_load = pd.DataFrame(temp_cur_load, columns=['current_charge'])
    temp_vol_load = pd.DataFrame(temp_vol_load, columns=['voltage_charge'])
    temp_time = pd.DataFrame(temp_time, columns=['time'])
    temp = temp_vol.join(temp_cur, how='left').join(temp_tem, how='left') \
        .join(temp_cur_load, how='left').join(temp_vol_load, how='left').join(temp_time, how='left')
    temp[['voltage', 'current', 'temperature', 'current_charge', 'voltage_charge', 'time']] \
        .to_csv(
        './data/charge_data/' + battery_name + '/charge_' + str(data.get('charge')) + '.csv',
        index=False,
        header=['voltage', 'current', 'temperature', 'current_charge', 'voltage_charge', 'time'])


# 将电池充电数据中可用特征存入csv
def charge_battery_data_to_csv(data):
    '''
        :param data:    data=get_charge_data()
        :return: 将充电数据--特征数据存入csv中
        '''
    num = 0
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        charge = []
        tem = []
        time = []
        for i in range(0, len(data[num].get(battery_name))):
            charge.append(data[num].get(battery_name)[i].get('charge'))
            tem.append(data[num].get(battery_name)[i].get('temperature'))
            time.append(data[num].get(battery_name)[i].get('time'))
        num += 1
        charge = pd.DataFrame(charge, columns=['charge'])
        tem = pd.DataFrame(tem, columns=['temperature'])
        time = pd.DataFrame(time, columns=['time'])
        temp = charge.join(tem, how='left').join(time, how='left')
        temp[['charge', 'temperature', 'time']] \
            .to_csv('./data/' + battery_name + '_charge.csv', index=False,
                    header=['charge', 'temperature', 'time'])


# 对电池充电特征数据进行扩展
def slide_window_to_extend_charge_data(data):
    '''
        :param data: pd.DataFrame格式，charge、temperature、time
            将扩展以后的数据写入csv
        :return:
        '''
    num = 5
    for b_data in data:
        b_data = b_data.drop(b_data.index[b_data['charge'] == 33])
        b_data = b_data.drop(b_data.index[b_data['charge'] == 170])
        temp_data = b_data[0:117]
        for i in range(1, 51):
            temp_data = pd.concat([temp_data, b_data[i:i + 117]], axis=0)

        temp_data[['charge', 'temperature', 'time']] \
            .to_csv('./data/extend_data/b_0' + str(num) + '_charge.csv',
                    index=False, header=['charge', 'temperature', 'time'])
        num += 1


'''对电池充电数据的处理---结束'''

# 将电池特征数据数据存入csv
# def battery_data_to_csv():
#     # 获取电池充电数据
#     data = get_charge_data()
#     num = 0
#     for name in battery_path:
#         battery_name = name.split('.')[0].split('\\')[-1]
#         charge = []
#         time = []
#         # 获取电池放电特征数据
#         data_dis = pd.read_csv('./data/' + battery_name + '_discharge.csv')
#         # 获取电池充电的特征数据
#         for i in range(0, len(data[num].get(battery_name)) - 2):
#             charge.append(data[num].get(battery_name)[i].get('charge'))
#             time.append(data[num].get(battery_name)[i].get('time'))
#         num += 1
#         charge = pd.DataFrame(charge, columns=['charge'])
#         time = pd.DataFrame(time, columns=['charge_time'])
#         temp = data_dis.join(charge, how='left').join(time, how='left')
#         # 将所有特征数据存入csv，以电池名称命名
#         temp[['discharge', 'temperature', 'time', 'capacity', 'soh', 'ic', 'charge', 'charge_time']] \
#             .to_csv('./data/' + battery_name + '.csv', index=False,
#                     header=['discharge', 'temperature', 'time', 'capacity', 'soh', 'ic', 'charge', 'charge_time'])


if __name__ == '__main__':
    # 放电数据获取和处理
    # data = get_data()
    # discharge_battery_data_to_csv(data)
    # data_to_csv(data)
    # # 获取数据
    # b_05 = pd.read_csv('./data/B0005.csv')
    # b_06 = pd.read_csv('./data/B0006.csv')
    # b_07 = pd.read_csv('./data/B0007.csv')
    # Standard_data_to_csv([b_07, b_06, b_05])
    # slide_window_to_extend_data([b_05, b_06])
    slide_window_to_extend_IC_data()

    # 充电数据获取和处理
    # charge_data = get_charge_data()
    # charge_battery_data_to_csv(charge_data)
    # b_05 = pd.read_csv('./data/B0005_charge.csv')
    # b_06 = pd.read_csv('./data/B0006_charge.csv')
    # slide_window_to_extend_charge_data([b_05, b_06])
    # battery_data_to_csv()
