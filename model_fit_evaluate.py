"""
作者：杨文豪

描述：创建LSTM模型并进行训练和评估


时间：2022/4/4 9:16
"""
import tensorflow as tf
from tensorflow.keras import optimizers, losses, Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 50
Lstm_model = Sequential([
    LSTM(units=200, return_sequences=True, dropout=0.2),
    LSTM(units=100, return_sequences=True, dropout=0.2),
    LSTM(units=100, return_sequences=True, dropout=0.2),
    LSTM(units=50, return_sequences=True, dropout=0.2),
    LSTM(units=50, dropout=0.2),
    Dense(1, activation='linear')
])
Lstm_model.build(input_shape=[None, 4, 1])
Lstm_model.summary()
Lstm_model.compile(optimizer=optimizers.RMSprop(0.01), loss=losses.mse, metrics=['accuracy'])

# 获取数据
b_05 = pd.read_csv('./data/B0005.csv')
b_06 = pd.read_csv('./data/B0006.csv')
b_07 = pd.read_csv('./data/B0007.csv')
# b_18 = pd.read_csv('./data/B0018.csv')

'''使用StandardScaler进行标准化，再去训练'''
data_train = pd.concat([b_05, b_06], axis=0)
feature=['discharge', 'voltage', 'temperature', 'time']
x_train = data_train[['discharge', 'voltage', 'temperature', 'time']]
y_train = data_train['soh']

x_test = b_07[['discharge', 'voltage', 'temperature', 'time']]
y_test = b_07['soh']
# 对数据进行标准化
tranfer = StandardScaler()
x_train = tranfer.fit_transform(x_train)
x_test = tranfer.fit_transform(x_test)
y_train = tranfer.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = tranfer.fit_transform(np.array(y_test).reshape(-1, 1))

# x_train=tf.data.Dataset.from_tensor_slices((x_train,y_train))
# sample=next(iter(x_train))
# print(sample.shape)
# 数据变形
x_train = tf.reshape(x_train, (len(x_train['discharge']), 4, 1))
x_test = tf.reshape(x_test, (len(x_test['discharge']), 4, 1))

# 训练
Lstm_model.fit(x_train, y_train, batch_size=8, epochs=200, verbose=1)
pred = Lstm_model.predict(x_test)
# 反标准化
pred = tranfer.inverse_transform(pred)
# 使用05，06的数据训练，使用07的数据去预测
# 画图，查看预测值与真实值之间的区别
font_color = ['r-', 'g-']
plt.rcParams["font.family"] = "Kaiti"
plt.title('SOH随循环次数的变化')
plt.xlabel('循环次数')
plt.ylabel('SOH(%)')
plt.plot(b_07['discharge'], b_07['soh'], 'r-', label='真实值')
plt.plot(b_07['discharge'], pred, 'g-', label='预测值')
plt.legend()
# B07电池SOH随循环变化图--batch_size=8，dropout=0.2
# B07电池SOH随循环变化图1--batch_size=4，dropout=0.5
plt.savefig("./image/pre_result/B07电池SOH随循环变化图1.jpg")
plt.show()
