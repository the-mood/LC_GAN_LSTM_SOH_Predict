"""
作者：杨文豪

描述：

时间：2022/4/8 9:18
"""
import tensorflow as tf
from tensorflow.keras import optimizers, losses, Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

b_names = ['B0005', 'B0006', 'B0007']
# ger_data = pd.read_csv("D:/Downloads/generator_data_8600.cvs")
real_data_b05 = pd.read_csv("./data/extend_data/b_05.csv")
real_data_b06 = pd.read_csv("./data/extend_data/b_06.csv")
real_data = pd.concat([real_data_b05, real_data_b06], axis=0)
c = ['r-', 'g-', 'b-']


def draw(g_data, r_data):
    plt.rcParams["font.family"] = "Kaiti"
    plt.xlabel('时间')
    plt.ylabel('温度')
    j = 0
    for i in range(0, 102):
        plt.title('第' + str(i) + '段数据中温度随循环次数的变化')
        plt.plot(r_data['discharge'][j:j + 117], g_data['temperature'][j:j + 117], 'r-', label='生成数据')
        plt.legend()
        plt.plot(r_data['discharge'][j:j + 117], r_data['temperature'][j:j + 117], 'g-', label='真实数据')
        plt.legend()
        plt.savefig('./image/生成数据和真实数据对比图/generator_data_8600/第' + str(i) + '段数据中温度随循环次数的变化')
        plt.show()
        j += 117


def test():
    plt.title('ic_max')
    plt.xlabel('循环次数')
    plt.ylabel('ic_max')
    num = 0
    for b_name in b_names:
        data = pd.read_csv('./data/all_feature_data/' + b_name + '_all_feature.csv')
        # data = pd.read_csv('./data/特征数据/' + b_name + '_ic峰值.csv')
        plt.plot(range(1, 169), data['ic_max'], c[num], label=b_name)
        plt.legend()
        num += 1
    plt.show()
    print()


if __name__ == '__main__':
    # result = st.pearsonr(list(ger_data['temperature']), list(real_data['temperature']))
    # print(result)
    # draw(ger_data, real_data)
    test()
