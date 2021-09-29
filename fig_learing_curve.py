import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
 
def moving_average(a, n=00) :
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
        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs)

    # path = dir_path + fileList[0]
    # res = np.load(path)
    # temp_rs = np.array(res['arr_0'])
    # avg_rs.append(temp_rs)    
    avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],10)
    return avg_rs

# ddpg_reward_S = output_avg('train_S_ddpg_sigma0_02_rate3_lane2/')
ddpg_reward_M = output_avg('train_M_ddpg_sigma0_02_rate3_lane2/')


# figure, ax = plt.subplots(figsize=(7.5, 6))
fig = plt.figure(figsize=(6, 4.5))


# plt.plot(range(ddpg_reward_S.shape[0]), ddpg_reward_S, color='0.25' )
plt.plot(range(ddpg_reward_M.shape[0]), ddpg_reward_M, color='#ff7f0e')
plt.legend()

plt.grid(linestyle=':')
plt.legend()
plt.ylabel("Reward")
plt.xlabel("Episode Index")
plt.show()

