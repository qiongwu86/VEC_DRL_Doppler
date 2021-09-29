import os
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  # 导入FontProperties

def read_log(dir_path):
    fileList = os.listdir(dir_path) #列出文件夹下所有的目录与文件
    fileList = [name for name in fileList if '.npz' in name]

    avg_step_reward = []
    avg_step_sum_power = []
    avg_step_buffer = []
    avg_step_offload_power = []
    avg_step_local_power = []

    for name in fileList[:]:
        path = dir_path + name
        res = np.load(path)

        temp_r = np.array(res['arr_0'])
        temp_sp = np.array(res['arr_1'])
        temp_b = np.array(res['arr_2'])
        temp_op = np.array(res['arr_3'])
        temp_lp = np.array(res['arr_4'])

        avg_step_reward.append(temp_r)
        avg_step_sum_power.append(temp_sp)
        avg_step_buffer.append(temp_b)
        avg_step_offload_power.append(temp_op)
        avg_step_local_power.append(temp_lp)

    avg_step_reward = moving_average(np.mean(avg_step_reward, axis=0, keepdims=True)[0])
    avg_step_sum_power = moving_average(np.mean(avg_step_sum_power, axis=0, keepdims=True)[0])
    avg_step_buffer = moving_average(np.mean(avg_step_buffer, axis=0, keepdims=True)[0])
    avg_step_offload_power = moving_average(np.mean(avg_step_offload_power, axis=0, keepdims=True)[0])
    avg_step_local_power = moving_average(np.mean(avg_step_local_power, axis=0, keepdims=True)[0])

    return avg_step_reward, avg_step_sum_power, avg_step_buffer, avg_step_offload_power, avg_step_local_power

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

[step_r, step_p, step_b, step_lp, step_op]= read_log('test_S_ddpg_sigma0_02_rate3_lane2/step_result/')

font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=22) 
font1 = {'family' : 'SimSun',
'weight' : 'normal',
'size'   : 22,
}

figure, ax = plt.subplots(figsize=(7.5, 6))

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=22)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.plot(range(step_lp.shape[0]), step_lp, color='0.75', label="本地计算功率", lw=1 )
plt.plot(range(step_op.shape[0]), step_op, color='0.25', label="任务卸载功率", lw=1 )
plt.legend(prop=font1)

plt.ylabel("功率", fontproperties=font)
plt.xlabel("时隙数\n(a)", fontproperties=font)
plt.grid(linestyle=':')
plt.show()