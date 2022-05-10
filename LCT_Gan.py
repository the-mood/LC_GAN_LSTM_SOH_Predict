"""
作者：杨文豪

描述：创建LSTM_Convelution_GAN网络，对电池数据进行生成

时间：2022/4/5 19:08
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Conv1DTranspose, Flatten
import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = Conv1D(512, kernel_size=3, activation=tf.nn.leaky_relu)
        self.conv2 = Conv1D(256, kernel_size=3, activation=tf.nn.leaky_relu)
        self.conv3 = Conv1D(117, kernel_size=3, activation=tf.nn.leaky_relu)
        self.lstm1 = LSTM(128, return_sequences=True, dropout=0.2)
        self.lstm2 = LSTM(256, return_sequences=True, dropout=0.2)
        self.lstm3 = LSTM(512, return_sequences=True, dropout=0.2)
        self.dense1 = Dense(128)
        self.dense2 = Dense(256)
        self.dense3 = Dense(512)
        self.dense = Dense(1)
        self.flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.dense1(inputs))
        x = tf.nn.leaky_relu(self.dense2(x))
        x = tf.nn.leaky_relu(self.dense3(x))
        x = tf.reshape(x, (2, 8, -1))
        x = tf.nn.leaky_relu(self.lstm1(x))
        x = tf.nn.leaky_relu(self.lstm2(x))
        x = tf.nn.leaky_relu(self.lstm3(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = tf.reshape(x, (2, 117, -1))
        out = self.dense(x)
        return out


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lstm1 = LSTM(128, return_sequences=True, dropout=0.2)
        self.lstm2 = LSTM(256, return_sequences=True, dropout=0.2)
        self.lstm3 = LSTM(512, return_sequences=True, dropout=0.2)
        self.dconv1 = Conv1DTranspose(256, kernel_size=3, activation=tf.nn.leaky_relu)
        self.dconv2 = Conv1DTranspose(128, kernel_size=3, activation=tf.nn.leaky_relu)
        self.dconv3 = Conv1DTranspose(64, kernel_size=3, activation=tf.nn.leaky_relu)
        self.dense1 = Dense(512)
        self.dense2 = Dense(256)
        self.dense3 = Dense(117)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.lstm1(inputs))
        x = tf.nn.leaky_relu(self.lstm2(x))
        x = tf.nn.leaky_relu(self.lstm3(x))
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.dense4(x)
        return out


# 计算判别器的误差函数
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 采样生成的数据
    fake_data = generator(batch_z, is_training)
    # 判定生成的数据
    d_fake_logits = discriminator(fake_data, is_training)
    # 判定真实的数据
    d_real_logits = discriminator(batch_x, is_training)
    # 真实数据与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成数据与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss


# 计算属于与标签为1的交叉熵
def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    # y = tf.ones_like(logits)
    # loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


# 计算属于与便签为0的交叉熵
def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    # y = tf.zeros_like(logits)
    # loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


# 计算生成器的误差函数
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_data = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_data, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)

    return loss


def test():
    g = Generator()
    d = Discriminator()
    x = tf.random.normal([2, 117, 1])
    z = tf.random.normal([2, 100])
    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)
    print(x_hat)


if __name__ == '__main__':
    test()
