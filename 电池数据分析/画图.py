"""
作者：杨文豪

描述：

时间：2022/5/3 9:58
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Kaiti"

data = pd.read_excel('./Ev_dataset.xlsx')
# print(data)
fea = ['velocity', 'SoC', 'voltage', 'current', 'maximum voltage', 'minimum voltage',
       'max. temperature', 'min. tem.', 'voltage of ES', 'current of ES']


def draw():
    for x_name in fea:
        num = 1
        for y_name in fea[num:]:
            plt.title(y_name + '随着' + x_name + '的变化')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.plot(data[x_name][1:], data[y_name][1:], 'ro')
            plt.savefig('./image/' + y_name + '随着' + x_name + '的变化.jpg')
            num += 1
            if num == 10:
                break


def test():
    num = 1
    for x_name in fea:
        for y_name in fea[num:]:
            plt.title(y_name + '随着' + x_name + '的变化')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.scatter(data[x_name][1:], data[y_name][1:], s=10)
            plt.savefig('./image/' + y_name + '随着' + x_name + '的变化.jpg')
            plt.show()
            if num == 10:
                break
        num += 1


if __name__ == '__main__':
    test()