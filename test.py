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

ger_data = pd.read_csv("D:/Downloads/generator_data_9200.cvs")
real_data_b05 = pd.read_csv("./data/extend_data/b_05.csv")
real_data_b06 = pd.read_csv("./data/extend_data/b_06.csv")
real_data = pd.concat([real_data_b05, real_data_b06], axis=0)


def draw(g_data, r_data):
    plt.rcParams["font.family"] = "Kaiti"
    plt.xlabel('时间')
    plt.ylabel('电压')
    j = 0
    for i in range(0, 102):
        plt.title('第' + str(i) + '段数据中电压随循环次数的变化')
        plt.plot(r_data['discharge'][j:j + 117], g_data['voltage'][j:j + 117], 'r-', label='生成数据')
        plt.legend()
        plt.plot(r_data['discharge'][j:j + 117], r_data['voltage'][j:j + 117], 'g-', label='真实数据')
        plt.legend()
        plt.show()
        j += 117


def test():
    data =pd.read_csv('./data/B0005_discharge.csv')
    plt.rcParams["font.family"] = "Kaiti"
    plt.title('增量容量曲线')
    plt.ylabel(data['ic'])
    plt.xlim(2.5, 3.85)


if __name__ == '__main__':
    # draw(ger_data, real_data)
    test()