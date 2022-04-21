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
data=pd.read_csv('./data/B0005.csv')
lis = data['discharge']



if len(os.listdir('./model'))==0:
    print(2)
else:
    print(1)
