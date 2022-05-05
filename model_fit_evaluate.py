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



# 获取数据
b_05_dis = pd.read_csv('./data/B0005_discharge.csv')
b_05_charge = pd.read_csv('./data/B0005_charge.csv')

b_06_dis = pd.read_csv('./data/B0006_discharge.csv')
b_06_charge = pd.read_csv('./data/B0006_charge.csv')

b_07_dis = pd.read_csv('./data/B0007_discharge.csv')
b_07_charge = pd.read_csv('./data/B0007_charge.csv')
# b_18 = pd.read_csv('./data/B0018.csv')
# 去除两个异常值
b_05_charge.drop(b_05_charge.index[b_05_charge['charge'] == 33], inplace=True)
b_05_charge.drop(b_05_charge.index[b_05_charge['charge'] == 170], inplace=True)
b_05_charge = b_05_charge.drop(labels=['temperature', 'charge'], axis=1)
b_05_charge = b_05_charge.reset_index(drop=True)

b_06_charge.drop(b_06_charge.index[b_06_charge['charge'] == 33], inplace=True)
b_06_charge.drop(b_06_charge.index[b_06_charge['charge'] == 170], inplace=True)
b_06_charge = b_06_charge.drop(labels=['temperature', 'charge'], axis=1)
b_06_charge = b_06_charge.reset_index(drop=True)

b_05 = b_05_dis.join(b_05_charge, how='left')
b_06 = b_06_dis.join(b_06_charge, how='left')
data_train = pd.concat([b_05, b_06], axis=0)

b_07_charge.drop(b_07_charge.index[b_07_charge['charge'] == 33], inplace=True)
b_07_charge.drop(b_07_charge.index[b_07_charge['charge'] == 170], inplace=True)
b_07_charge = b_07_charge.reset_index(drop=True)

b_07 = b_07_dis.join(b_07_charge['charge_time'], how='left')
'''使用StandardScaler进行标准化，再去训练'''
feature = ['temperature', 'time', 'ic', 'charge_time']
x_train = data_train[['temperature', 'discharge_time', 'ic', 'charge_time']]
y_train = data_train['soh']

x_test = b_07[['temperature', 'discharge_time', 'ic', 'charge_time']]
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
x_train = tf.reshape(x_train, (len(data_train['ic']), 4, 1))
x_test = tf.reshape(x_test, (len(b_07['ic']), 4, 1))


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
Lstm_model.compile(optimizer=optimizers.RMSprop(0.005), loss=losses.mse, metrics=['accuracy'])
# 训练
Lstm_model.fit(x_train, y_train, batch_size=8, epochs=300, verbose=1)
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
plt.savefig('./image/pred_result/B07电池SOH随循环变化图3.jpg')
plt.show()
