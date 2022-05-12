"""
作者：杨文豪

描述：

时间：2022/5/3 11:22
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as st
from scipy.signal import savgol_filter

# base_path = 'D:/当前可能用到的文件/生成的数据/discharge_time'
# base_path = 'D:/当前可能用到的文件/生成的数据/生成数据保存/discharge_time'
# base_path = 'D:/当前可能用到的文件/生成的数据/生成数据保存/discharge_to_min_voltage_time'
# base_path = 'D:/当前可能用到的文件/生成的数据/生成数据保存/CC_ratio'
# base_path = 'D:/当前可能用到的文件/生成的数据/生成数据保存/charge_to_4.2v_time'
base_path = 'D:/当前可能用到的文件/生成的数据/生成数据保存/ic_max'

y_name = base_path.split('/')[-1]
file_names = os.listdir(base_path)
file_names.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
real_data_b05 = pd.read_csv("./data/extend_data/b_005_extend_all_feature.csv")
real_data_b06 = pd.read_csv("./data/extend_data/b_006_extend_all_feature.csv")
real_data = pd.concat([real_data_b05, real_data_b06], axis=0)
plt.rcParams["font.family"] = "Kaiti"


# 画出生成的数据和真实的数据之间的对比图
def draw_g_data_to_r_data():
    temp = 0
    data = ''
    for file_name in file_names:
        ger_data = pd.read_csv(base_path + '/' + file_name)
        result = st.pearsonr(list(ger_data[y_name]), list(real_data[y_name]))
        if result[0] > temp:
            temp = result[0]
            data = file_name
        # if result[0] > 0.85:
        #     print('皮尔逊相关系数：' + str(result[0]) + '   对应文件为：' + file_name)
        plt.xlabel('循环次数')
        plt.ylabel(y_name)
        j = 0
        file = file_name.split('.')[0]
        if not os.path.exists('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file):
            os.mkdir('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file)
        with open('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file + '/readme.txt',
                  'w', encoding='utf-8') as f:
            # 判断每一段的生成数据和真实数据之间的皮尔逊相关系数
            for i in range(0, 102):
                p = st.pearsonr(ger_data[y_name][j:j + 117], real_data[y_name][j:j + 117])
                if p[0] > 0.99:
                    # 将皮尔逊系数大于0.99的每一段数据记录在reademe文件中
                    f.write(file_name + '中第' + str(i) + '段数据,索引为:' + str(j) + '~' + str(j + 117) +
                            ',皮尔逊相关系数为：' + str(p[0]) + '\n')
                    print(file_name + '中第' + str(i) + '段数据,索引为:' + str(j) + '~' + str(j + 117) +
                          ',皮尔逊相关系数为：' + str(p[0]))
                # 画图,画出每一段生成数据和真实数据之间的对比图
                plt.title('第' + str(i) + '段数据中' + y_name + '随循环次数的变化')
                plt.plot(real_data['discharge'][j:j + 117], ger_data[y_name][j:j + 117], 'r-', label='生成数据')
                plt.legend()
                plt.plot(real_data['discharge'][j:j + 117], real_data[y_name][j:j + 117], 'g-', label='真实数据')
                plt.legend()
                plt.savefig('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file +
                            '/第' + str(i) + '段数据中' + y_name + '随循环次数的变化.jpg')
                plt.show()
                j += 117
        print(file_name + "对比完成,皮尔逊相关系数为：" + str(result))
    print('皮尔逊相关系数最高为：' + str(temp) + '     对应文件为：' + data)


# 计算皮尔逊相关系数,找出皮尔逊相关系数最大的文件
def test_pierxun():
    temp = 0
    data = ''
    for file_name in file_names:
        ger_data = pd.read_csv(base_path + '/' + file_name)
        result = st.pearsonr(list(ger_data[y_name]), list(real_data[y_name]))
        if result[0] > temp:
            temp = result[0]
            data = file_name
        if result[0] > 0.85:
            print('皮尔逊相关系数：' + str(result[0]) + '   对应文件为：' + file_name)

        j = 0
        file = file_name.split('.')[0]
        for i in range(0, 102):
            p = st.pearsonr(ger_data[y_name][j:j + 117], real_data[y_name][j:j + 117])
            if p[0] > 0.99:
                if not os.path.exists('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file):
                    os.mkdir('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file)
                with open('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file + '/readme.txt',
                          'w', encoding='utf-8') as f:
                    f.write(file_name + '中第' + str(i) + '段数据,索引为:' + str(j) + '~' + str(j + 117) +
                            ',皮尔逊相关系数为：' + str(p[0]) + '\n')
                print(file_name + '中第' + str(i) + '段数据,索引为:' + str(j) + '~' + str(j + 117) +
                      ',皮尔逊相关系数为：' + str(p[0]))
            j += 117

    print('皮尔逊相关系数最高为：' + str(temp) + '     对应文件为：' + data)


# 选中某一个生成的数据，将生成的数据修正后存入csv     y_name=feature_name
def change_generator_data(feature_name):
    # ger_data = pd.read_csv('./data/generator_data/' + feature_name + '/generator_data_28800.cvs')
    ger_data = pd.read_csv('./data/generator_data/'+feature_name+'/generator_data_9200.cvs')
    plt.xlabel('循环次数')
    plt.ylabel(feature_name)
    j = 0
    temp = pd.DataFrame(range(0, 117), columns=['index'])
    for i in range(0, 102):
        p = st.pearsonr(ger_data[feature_name][j:j + 117], real_data[feature_name][j:j + 117])
        if p[0] > 0.99:
            result = savgol_filter(ger_data[feature_name][j:j + 117], 3, 1, mode='nearest')
            t = real_data[feature_name][j:j + 117] - result
            # 大于50是电池6号的数据
            if i > 50:
                # 为存储文件做准备
                tem = pd.DataFrame(result + np.mean(t),
                                   columns=['b6_' + str((i - 50) % 117) + '_' + str((i - 50) % 117 + 117)])
                temp = pd.concat([temp, tem], axis=1)
                # 画图
                plt.title('b6第' + str(i - 50) + '段数据修正后' + feature_name + '随循环次数的变化')
                plt.plot(real_data['discharge'][j:j + 117], result + np.mean(t), 'r-', label='生成数据')
                plt.legend()
                plt.plot(real_data['discharge'][j:j + 117], real_data[feature_name][j:j + 117], 'g-', label='真实数据')
                plt.legend()
                plt.savefig('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + feature_name +
                            '/b6第' + str(i - 51) + '段数据修正后' + feature_name + '随循环次数的变化.jpg')
                plt.show()
            # 小于50是电池5号的数据
            else:
                tem = pd.DataFrame(result + np.mean(t),
                                   columns=['b5_' + str(i % 117) + '_' + str(i % 117 + 117)])
                temp = pd.concat([temp, tem], axis=1)

                plt.title('b5第' + str(i) + '段数据修正后' + feature_name + '随循环次数的变化')
                plt.plot(real_data['discharge'][j:j + 117], result + np.mean(t), 'r-', label='生成数据')
                plt.legend()
                plt.plot(real_data['discharge'][j:j + 117], real_data[feature_name][j:j + 117], 'g-', label='真实数据')
                plt.legend()
                plt.savefig('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + feature_name +
                            '/b5第' + str(i) + '段数据修正后' + feature_name + '随循环次数的变化.jpg')
                plt.show()
        j += 117
    temp = temp.round(decimals=2)
    temp.to_csv('./data/可用的生成数据/' + feature_name + '/修正后的生成数据1.csv', index=False)
    print()


# 制作新的数据集       y_name=feature_name
def make_new_data(feature_name, num):
    data = pd.read_csv('./data/可用的生成数据/' + feature_name + '/修正后的生成数据.csv')
    columns = data.columns
    # 选择某一列作为要制作的新的数据集  修改columns的下标即可修改生成数据的内容
    b_num = columns[num].split('_')[0].split('b')[-1]
    index = columns[num].split('_')[1]
    r_data = pd.read_csv('./data/all_feature_data/B000' + b_num + '_all_feature.csv')
    new_data = r_data[feature_name][:int(index)].append(data[columns[num]], ignore_index=True) \
        .append(r_data[feature_name][int(index) + 117:])
    new_data.to_csv('./data/可用的生成数据/' + feature_name + '/训练可用数据' + b_num + '.csv',
                    index=False, header=[feature_name])
    return b_num


# 比较生成的训练数据和真实的数据
def compare_new_data_to_real_data(feature_name, num):
    new_data = pd.read_csv('./data/可用的生成数据/' + feature_name + '/训练可用数据' + num + '.csv')
    r_data = pd.read_csv('./data/all_feature_data/B000' + num + '_all_feature.csv')
    # r_data = pd.read_csv('./data/all_feature_data/B0006_all_feature.csv')
    plt.title('新生成的数据和真实数据对比')
    plt.xlabel('循环次数')
    plt.ylabel(feature_name)
    plt.plot(range(1, 169), r_data[feature_name], 'g-', label='真实数据')
    plt.legend()
    plt.plot(range(1, 169), new_data[feature_name], 'r-', label='生成数据')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_g_data_to_r_data()
    # test_pierxun()
    # change_generator_data(y_name)
    # n = make_new_data(y_name, 2)
    # compare_new_data_to_real_data(y_name, n)
