"""
作者：杨文豪

描述：GAN训练生成电池放电时IC容量变化


时间：2022/4/15 9:19
"""
import tensorflow as tf
from tensorflow.keras import optimizers
from GAN import Generator, Discriminator, d_loss_fn, g_loss_fn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

b05 = pd.read_csv('./data/extend_data/b_005_extend_all_feature.csv')
b06 = pd.read_csv('./data/extend_data/b_006_extend_all_feature.csv')
transfer = StandardScaler()


def make_gan_dataset():
    dataset = []
    for b in [b05, b06]:
        i = 0
        while i < 5851:
            dataset.append(list(b['ic_max'][i:i + 117]))
            i += 117
    for i in range(0, len(dataset)):
        dataset[i] = transfer.fit_transform(np.array(dataset[i]).reshape(-1, 1))
    data_set = tf.data.Dataset.from_tensor_slices(dataset)
    print(len(dataset[0]), len(dataset))
    sample = next(iter(data_set))
    # (117，1)
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample))
    print(data_set)
    return data_set


def train_for_generator_ic(dataset):
    tf.random.set_seed(2222)
    np.random.seed(2222)
    z_dim = 100  # 隐藏向量z的长度
    epochs = 30000  # 训练步数
    batch_size = 64
    learning_rate = 0.001
    is_training = True
    # 无限制的从ds中拿取数据，直到epoch训练完
    dataset = dataset.repeat()
    db_iter = iter(dataset)
    # 创建生成器和判别器
    if len(os.listdir('./model/ic_max')) != 1:
        generator = Generator()
        generator.build(input_shape=(None, z_dim))
        generator.load_weights('./model/ic_max/generator.ckpt')
        discriminator = Discriminator()
        discriminator.build(input_shape=(None, 117, 1))
        discriminator.load_weights('./model/ic_max/discriminator.ckpt')
    else:
        generator = Generator()
        generator.build(input_shape=(None, z_dim))
        discriminator = Discriminator()
        discriminator.build(input_shape=(None, 117, 1))
    # 创建优化器，两个优化器分别优化生成器和判别器
    g_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    d_losses, g_losses = [], []
    for epoch in range(epochs):
        # 训练鉴别器
        for _ in range(1):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            # 采样真实图片
            batch_x = next(db_iter)
            # 训练判别器
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        # 训练生成器
        # for _ in range(30):
        # 2. 训练生成器
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        # 生成器前向计算
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 200 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
            # 用当前的生成器生成数据
            z = tf.random.normal([102, z_dim])
            fake_data = generator(z, training=False)
            # 数据反标准化
            fake_data = fake_data.numpy()
            for k in range(0, len(fake_data)):
                fake_data[k] = transfer.inverse_transform(fake_data[k])
            # 将生成的数据转化为DataFrame格式方便存入csv
            g_data = pd.DataFrame()
            for i in range(0, 102):
                temp = pd.DataFrame(fake_data[i], columns=['ic_max'])
                g_data = pd.concat([g_data, temp], axis=0)
            # 将生成的数据存入csv
            g_data[['ic_max']] \
                .to_csv('./data/generator_data/ic_max/generator_data_%d.cvs' % epoch,
                        index=False, header=['ic_max'])

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        if epoch % 1000 == 1 and epoch != 1:
            # print(d_losses)
            # print(g_losses)
            generator.save_weights('./model/ic_max/generator.ckpt', )
            discriminator.save_weights('./model/ic_max/discriminator.ckpt')


if __name__ == '__main__':
    ds = make_gan_dataset()
    train_for_generator_ic(ds)
