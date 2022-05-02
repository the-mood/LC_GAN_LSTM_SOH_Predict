"""
作者：杨文豪

描述：最原始的GAN

时间：2022/4/8 10:56
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Dense(128)
        self.fc2 = Dense(256)
        self.bn1 = BatchNormalization()

        self.fc3 = Dense(512)
        self.bn2 = BatchNormalization()

        self.fc4 = Dense(256)
        self.bn3 = BatchNormalization()

        self.fc5 = Dense(117)

    def call(self, inputs, training=None, mask=None):
        # [b,inputs]=>[b,128]
        x = tf.nn.leaky_relu(self.fc1(inputs))
        # [b,128]=>[b,256]
        x = tf.nn.leaky_relu(self.bn1(self.fc2(x), training=training))
        # [b,256]=>[b,512]
        x = tf.nn.leaky_relu(self.bn2(self.fc3(x), training=training))
        # [b,512]=>[b,256]
        x = tf.nn.leaky_relu(self.bn3(self.fc4(x), training=training))
        # [b,256]=>[b,468]
        x = self.fc5(x)
        out = tf.reshape(x, (-1, 117, 1))
        # out = tf.nn.tanh(x)
        return out


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = Dense(128)
        self.fc2 = Dense(256)
        self.fc3 = Dense(512)
        self.fc4 = Dense(1)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, (-1, 117))
        x = tf.nn.leaky_relu(self.fc3(inputs))
        x = tf.nn.leaky_relu(self.fc2(x))
        x = tf.nn.leaky_relu(self.fc1(x))
        out = self.fc4(x)
        # out = tf.nn.sigmoid(x)
        return out


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
