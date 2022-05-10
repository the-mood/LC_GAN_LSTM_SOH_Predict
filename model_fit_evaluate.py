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
b_05 = pd.read_csv('./data/all_feature_data/B0005_all_feature.csv')
b_06 = pd.read_csv('./data/all_feature_data/B0006_all_feature.csv')
b_07 = pd.read_csv('./data/all_feature_data/B0007_all_feature.csv')
data_train = pd.concat([b_05, b_06], axis=0)

'''使用StandardScaler进行标准化，再去训练'''
feature = ['charge_to_4.2v_time', 'CC_ratio', 'discharge_to_min_voltage_time', 'ic_max', 'discharge_time']
x_train = data_train[feature]
y_train = data_train['soh']

x_test = b_07[feature]
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
x_train = tf.reshape(x_train, (len(data_train['ic_max']), 5, 1))
x_test = tf.reshape(x_test, (len(b_07['ic_max']), 5, 1))

# 50
Lstm_model = Sequential([
    LSTM(units=200, return_sequences=True, dropout=0.2),
    LSTM(units=100, return_sequences=True, dropout=0.2),
    LSTM(units=100, return_sequences=True, dropout=0.2),
    LSTM(units=50, return_sequences=True, dropout=0.2),
    LSTM(units=50, dropout=0.2),
    Dense(1, activation='linear')
])
Lstm_model.build(input_shape=[None, 5, 1])
Lstm_model.summary()
Lstm_model.compile(optimizer=optimizers.RMSprop(0.01), loss=losses.mse, metrics=['accuracy'])
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
plt.savefig('./image/pred_result/B07电池SOH随循环变化图3.jpg')
plt.show()
