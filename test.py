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

transfer=StandardScaler()
data=pd.read_csv('./data/generator_data/generator_data_0.cvs')
data=transfer.fit_transform(data[:117])
data=transfer.inverse_transform(data)
print()

