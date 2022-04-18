"""
作者：杨文豪

描述：

时间：2022/4/13 20:34
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from data_process import battery_path

plt.rcParams["font.family"] = "Kaiti"


# 画图--电池容量随循环次数的变化
def draw_fig_of_capacity(feature_name):
    # 图的配置
    num = 0
    font_color = ['r-', 'g-', 'b-', 'y-']
    plt.title(feature_name + '随循环次数的变化')
    plt.xlabel('循环次数')
    plt.ylabel(feature_name)

    for name in battery_path:
        battery_name = name.split('.')[0].split('\\')[-1]
        battery_data = pd.read_csv('./data/' + battery_name + '.csv')
        plt.plot(battery_data['discharge'], battery_data[feature_name], font_color[num], label=battery_name)
        plt.legend()
        num += 1
    plt.savefig("./image/feature/电池" + feature_name + "随循环变化图.jpg")
    plt.show()


# 可视化，某一次放电时电压，电流，温度，电流负载，电压负载随时间的变化
def draw_fig():
    data = pd.read_csv('./data/B0005/dis_1_cap_1.8564874208181574.csv')
    # 图的配置
    font_color = ['r-', 'g-', 'b-', 'y-']
    plt.rcParams["font.family"] = "Kaiti"
    plt.title('影响因素随时间的变化')
    plt.xlabel('时间')
    for name in data.columns[:-1]:
        plt.ylabel(name)
        plt.plot(data['time'], data[name], 'r-')
        plt.show()


if __name__ == '__main__':
    fea = ['voltage', 'temperature', 'time', 'capacity', 'soh']
    for name in fea:
        draw_fig_of_capacity(name)
