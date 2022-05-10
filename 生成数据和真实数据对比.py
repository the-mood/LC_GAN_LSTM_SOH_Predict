"""
作者：杨文豪

描述：

时间：2022/5/3 11:22
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as st

# base_path = 'D:/当前可能用到的文件/生成的数据/temperature'
base_path = 'D:/当前可能用到的文件/生成的数据/dis_time'
# base_path = 'D:/当前可能用到的文件/生成的数据/ic'
# base_path = 'D:/当前可能用到的文件/生成的数据/charge_time'
y_name = base_path.split('/')[-1]
file_names = os.listdir(base_path)
real_data_b05 = pd.read_csv("./data/extend_data/b_05.csv")
# real_data_b05 = pd.read_csv("./data/extend_data/b_05_charge.csv")
# real_data_b05 = pd.read_csv("./data/extend_data/b_05_dis_ic.csv")
real_data_b06 = pd.read_csv("./data/extend_data/b_06.csv")
# real_data_b06 = pd.read_csv("./data/extend_data/b_06_charge.csv")
# real_data_b06 = pd.read_csv("./data/extend_data/b_06_dis_ic.csv")
real_data = pd.concat([real_data_b05, real_data_b06], axis=0)


def draw_g_data_to_r_data():
    for file_name in file_names:
        ger_data = pd.read_csv(base_path + '/' + file_name)
        result = st.pearsonr(list(ger_data[y_name]), list(real_data['discharge_time']))
        plt.rcParams["font.family"] = "Kaiti"
        plt.xlabel('循环次数')
        plt.ylabel(y_name)
        j = 0
        file = file_name.split('.')[0]
        if not os.path.exists('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file):
            os.mkdir('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file)
        for i in range(0, 102):
            p = st.pearsonr(ger_data[y_name][j:j + 117], real_data['discharge_time'][j:j + 117])
            if p[0] > 0.99:
                print(file_name + '中第' + str(i) + '段数据，索引为:' + str(j) + '~' + str(j + 117) + '，皮尔逊相关系数为：' + str(p[0]))
            plt.title('第' + str(i) + '段数据中' + y_name + '随循环次数的变化')
            plt.plot(real_data['discharge'][j:j + 117], ger_data[y_name][j:j + 117], 'r-', label='生成数据')
            plt.legend()
            plt.plot(real_data['discharge'][j:j + 117], real_data['discharge_time'][j:j + 117], 'g-', label='真实数据')
            plt.legend()
            plt.savefig('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file +
                        '/第' + str(i) + '段数据中' + y_name + '随循环次数的变化.jpg')
            plt.show()
            j += 117
        print(file_name + "对比完成，皮尔逊相关系数为：" + str(result))


def test_pierxun():
    temp = 0
    data = ''
    for file_name in file_names:
        ger_data = pd.read_csv(base_path + '/' + file_name)
        result = st.pearsonr(list(ger_data[y_name]), list(real_data['discharge_time']))
        if result[0] > temp:
            temp = result[0]
            data = file_name
        if result[0] > 0.8:
            print('皮尔逊相关系数：' + str(result[0]) + '   对应文件为：' + file_name)

        j = 0
        file = file_name.split('.')[0]
        if not os.path.exists('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file):
            os.mkdir('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file)
        # for i in range(0, 102):
        #     p = st.pearsonr(ger_data[y_name][j:j + 117], real_data[y_name][j:j + 117])
        #     if p[0] > 0.95:
        #         print(file_name + '中第' + str(i) + '段数据，索引为:' + str(j) + '~' + str(j + 117) +
        #               '，皮尔逊相关系数为：' + str(p[0]))
        #     j += 117
        with open('D:/当前可能用到的文件/生成的数据/生成数据和真实数据对比图/' + y_name + '/' + file + '/readme.txt',
                  'w', encoding='utf-8') as f:
            for i in range(0, 102):
                p = st.pearsonr(ger_data[y_name][j:j + 117], real_data['discharge_time'][j:j + 117])
                if p[0] > 0.99:
                    f.write(file_name + '中第' + str(i) + '段数据，索引为:' + str(j) + '~' + str(j + 117) +
                            '，皮尔逊相关系数为：' + str(p[0]) + '\n')
                    print(file_name + '中第' + str(i) + '段数据，索引为:' + str(j) + '~' + str(j + 117) +
                          '，皮尔逊相关系数为：' + str(p[0]))
                j += 117

    print('皮尔逊相关系数最高为：' + str(temp) + '     对应文件为：' + data)


if __name__ == '__main__':
    # draw_g_data_to_r_data()
    test_pierxun()
