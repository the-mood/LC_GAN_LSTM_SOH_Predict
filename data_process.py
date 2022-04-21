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

base_path = r'D:\当前可能用到的文件\电池数据集\NASA_data'
f_name = 'BatteryAgingARC-FY08Q4'
# 获取当前路径下所有的.mat后缀的文件
battery_path = glob.glob(os.path.join(base_path + '\\' + f_name, '*.mat'))
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


# 对电池的数据进行处理
def process_data(file_name, b_name):
    '''
    :param file_name: 电池数据所在的文件名
    :param b_name: 电池名称
        将每次放电的数据存入csv中
            数据包括：电压，电流，温度，电流负载，电压负载，时间
    :return:
    '''
    if not os.path.exists('./data/' + b_name):
        os.mkdir('./data/' + b_name)
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
            data_to_csv(temp_dict, b_name)

            max_tem = np.max(temp_dict.get('temperature'))
            vol = temp_dict.get('voltage')[-1]
            print(max_tem, vol)
            t_dict = {'discharge': discharge_Num,
                      'voltage': np.array(data_B[0][0][0][0][i][3][0][0][0][0])[-1],
                      'temperature': np.max(np.array(data_B[0][0][0][0][i][3][0][0][2][0])),
                      'time': np.array(data_B[0][0][0][0][i][3][0][0][5][0])[-1],
                      'capacity': list(data_B[0][0][0][0][i][3][0][0][6][0])
                      }
            b_data.append(t_dict)
    return b_data


# 获取电池放电数据
def get_data():
    # 存放base_path下所有电池的循环次数和容量数据
    battery_data = []
    # 获取电池的具体名称
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        temp_data = {battery_name: process_data(f_name, battery_name)}
        battery_data.append(temp_data)
    '''
        battery_data=[dict0,dict1,dict2,dict3]
        dict0={'B0005': [{'discharge': 1, 'capacity': [1.8564874208181574]}, 
                {'discharge': 2, 'capacity': [1.846327249719927]},....]
    '''
    return battery_data


# 将获取的数据--循环次数，放电最后时刻的电压，最高温度，放电时间，容量存入csv文件中
def battery_data_to_csv(data):
    num = 0
    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        dis = []
        vol = []
        tem = []
        time = []
        cap = []
        soh = []
        for i in range(0, len(data[num].get(battery_name))):
            dis.append(data[num].get(battery_name)[i].get('discharge'))
            vol.append(data[num].get(battery_name)[i].get('voltage'))
            tem.append(data[num].get(battery_name)[i].get('temperature'))
            time.append(data[num].get(battery_name)[i].get('time'))
            cap.append(data[num].get(battery_name)[i].get('capacity')[0])
        for j in range(0, len(cap)):
            soh.append((cap[j] / cap[0]) * 100)
        num += 1
        dis = pd.DataFrame(dis, columns=['discharge'])
        vol = pd.DataFrame(vol, columns=['voltage'])
        tem = pd.DataFrame(tem, columns=['temperature'])
        time = pd.DataFrame(time, columns=['time'])
        cap = pd.DataFrame(cap, columns=['capacity'])
        soh = pd.DataFrame(soh, columns=['soh'])
        temp = dis.join(vol, how='left').join(tem, how='left').join(time, how='left').join(cap, how='left').join(soh,
                                                                                                                 how='left')
        temp[['discharge', 'voltage', 'temperature', 'time', 'capacity', 'soh']] \
            .to_csv('./data/' + battery_name + '.csv', index=False,
                    header=['discharge', 'voltage', 'temperature', 'time', 'capacity', 'soh'])


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
    temp_vol = pd.DataFrame(temp_vol, columns=['voltage'])
    temp_cur = pd.DataFrame(temp_cur, columns=['current'])
    temp_tem = pd.DataFrame(temp_tem, columns=['temperature'])
    temp_cur_load = pd.DataFrame(temp_cur_load, columns=['current_load'])
    temp_vol_load = pd.DataFrame(temp_vol_load, columns=['voltage_load'])
    temp_time = pd.DataFrame(temp_time, columns=['time'])
    temp = temp_vol.join(temp_cur, how='left').join(temp_tem, how='left') \
        .join(temp_cur_load, how='left').join(temp_vol_load, how='left').join(temp_time, how='left')
    temp[['voltage', 'current', 'temperature', 'current_load', 'voltage_load', 'time']] \
        .to_csv(
        './data/' + battery_name + '/dis_' + str(data.get('discharge')) + '_cap_' + str(
            data.get('capacity')[0]) + '.csv',
        index=False,
        header=['voltage', 'current', 'temperature', 'current_load', 'voltage_load', 'time'])


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


if __name__ == '__main__':
    # data = get_data()
    # battery_data_to_csv(data)
    # data_to_csv(data)
    # # 获取数据
    b_05 = pd.read_csv('./data/B0005.csv')
    b_06 = pd.read_csv('./data/B0006.csv')
    b_07 = pd.read_csv('./data/B0007.csv')
    Standard_data_to_csv([b_07, b_06, b_05])
    # slide_window_to_extend_data([b_05, b_06])
