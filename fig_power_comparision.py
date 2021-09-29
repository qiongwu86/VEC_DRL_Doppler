import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  


def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList[8:]:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_1'])
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
	return avg_rs


# ddpg_avg_power = np.mean(output_avg('test_S_ddpg_sigma0_02_rate3_lane2/'), axis=0)
# GD_local_avg_power = np.mean(output_avg('test_S_GD_Local_lane2/'), axis=0)
# GD_offload_avg_power = np.mean(output_avg('test_S_GD_Offload_lane2/'), axis=0)


ddpg_avg_power_m = np.mean(output_avg('test_M_ddpg_sigma0_02_rate3_lane2/'), axis=0)
GD_local_avg_power_m = np.mean(output_avg('test_M_GD_Local_lane2_rate_3/'), axis=0)
GD_offload_avg_power_m = np.mean(output_avg('test_M_GD_Offload_lane2_rate_3/'), axis=0)


# power = [ddpg_avg_power,GD_local_avg_power,GD_offload_avg_power]
power_M = [ddpg_avg_power_m, GD_local_avg_power_m, GD_offload_avg_power_m]
labels = ['DDPG', 'GD-Local', 'GD-Offload']

name = ["Policies"]
y1 = [ddpg_avg_power_m]
y2 = [GD_local_avg_power_m]
y3 = [GD_offload_avg_power_m]


x = np.arange(len(name))
width = 0.25

plt.bar(x, y1,  width=width, label='Optimal',color='#1f77b4')
plt.bar(x + width, y2, width=width, label='GD-Local', color='salmon', tick_label=name)
plt.bar(x + 2 * width, y3, width=width, label='GD-Offload', color='darkred')


plt.xticks()
plt.ylabel('Average power')
# plt.ylabel('Average buffer length')

plt.grid(linestyle=':')

plt.legend()
plt.show()