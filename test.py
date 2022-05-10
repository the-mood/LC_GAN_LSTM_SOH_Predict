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

# ger_data = pd.read_csv("D:/Downloads/generator_data_8600.cvs")
real_data_b05 = pd.read_csv("./data/extend_data/b_05.csv")
real_data_b06 = pd.read_csv("./data/extend_data/b_06.csv")
real_data = pd.concat([real_data_b05, real_data_b06], axis=0)


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
    d=real_data_b05.get('discharge')
    print()


if __name__ == '__main__':
    # result = st.pearsonr(list(ger_data['temperature']), list(real_data['temperature']))
    # print(result)
    # draw(ger_data, real_data)
    test()
