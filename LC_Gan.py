"""
作者：杨文豪

描述：创建LSTM_Convelution_GAN网络，对电池数据进行生成

时间：2022/4/5 19:08
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPool1D, BatchNormalization, Flatten
import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv1D(128, kernel_size=3, activation=tf.nn.relu)
        self.maxpool1 = MaxPool1D(2)




class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
