# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *


def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# avg_rs = moving_average(avg_rs[0],10)
def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) #列出文件夹下所有的目录与文件
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_3'])
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, axis=0, keepdims=True)
	# avg_rs = moving_average(avg_rs[0],10)
	avg_rs = avg_rs[0]
	return avg_rs




dir_path = 'test_S_ddpg_sigma0_02_rate2_lane2/step_result/'
fileList = os.listdir(dir_path) #列出文件夹下所有的目录与文件
fileList = [name for name in fileList if '.npz' in name]
avg_rs = []

path = dir_path + fileList[0]
res = np.load(path)
temp_rs = np.array(res['arr_0'])
avg_rs.append(temp_rs)
avg_rs = moving_average(avg_rs[0])
# avg_rs = avg_rs[0]


fig = plt.figure(figsize=(6, 4.5))
plt.plot(range(avg_rs.shape[0]), avg_rs, color='#1f77b4', label='DDPG')

plt.grid(linestyle=':')
plt.legend()
plt.ylabel('Average Reward Per Episode')
plt.xlabel('Episode Index')
plt.show()
