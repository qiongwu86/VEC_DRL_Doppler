#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import numpy as np
import ipdb as pdb
from helper import *


# In[7]:


def read_log(dir_path, user_idx=0, b=-1, e=-1):
    # name = []
    fileList = os.listdir(dir_path) #列出文件夹下所有的目录与文件
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    avg_ps = []
    avg_bs = []
    avg_os = []

    if b == -1:
        b = 0
        e = len(fileList)
    elif e == -1:
        e = b+1
    n=1    
    for name in fileList[b:e]:
        path = dir_path + name
        res = np.load(path)

        temp_rs = np.array(res['arr_0'])
        # avg_rs.append(temp_rs[:, user_idx])
        
        temp_ps = np.array(res['arr_1'])
        # avg_ps.append(temp_ps[:, user_idx])
        
        temp_bs = np.array(res['arr_2'])
        # avg_bs.append(temp_bs[:, user_idx])
        
        temp_os = np.array(res['arr_3'])
        # avg_os.append(temp_os[:, user_idx])
    
    avg_rs = np.mean(temp_rs, axis=0, keepdims=True)
    avg_ps = np.mean(temp_ps, axis=0, keepdims=True)
    avg_bs = np.mean(temp_bs, axis=0, keepdims=True)
    avg_os = np.mean(temp_os, axis=0, keepdims=True)
    
    return avg_rs, avg_ps, avg_bs, avg_os, name


# In[8]:


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[9]:


def plot_now(a1, a2, a11, a12, a21, a22, y_label, path):
    avg_a1 = moving_average(a1[0], win)
    avg_a2 = moving_average(a2[0], win)
    
    avg_a11 = moving_average(a11[0], win)
    avg_a12 = moving_average(a12[0], win)
    
    avg_a21 = moving_average(a21[0], win)
    avg_a22 = moving_average(a22[0], win)
    
    fig = plt.figure(figsize=(6, 4.5))
    # plt.plot(range(a1.shape[0]), a1)
    plt.plot(range(avg_a1.shape[0]), avg_a1, color='#1f77b4', label='DDPG, User 1')
    # plt.plot(range(a2.shape[0]), a2)
    plt.plot(range(avg_a2.shape[0])[:-1:5], avg_a2[:-1:5], color='#ff7f0e', label='DQN, User 1')
    
    plt.plot(range(avg_a11.shape[0])[:-1:5], avg_a11[:-1:5], color='#1f77b4', linestyle='-.', label='DDPG, User 2')
    plt.plot(range(avg_a12.shape[0])[:-1:5], avg_a12[:-1:5], color='#ff7f0e', linestyle='-.', label='DQN, User 2')
    
    plt.plot(range(avg_a21.shape[0])[:-1:5], avg_a21[:-1:5], color='#1f77b4', linestyle='--', label='DDPG, User 3')
    plt.plot(range(avg_a22.shape[0])[:-1:5], avg_a22[:-1:5], color='#ff7f0e', linestyle='--', label='DQN, Use 3')


    plt.grid(linestyle=':')
    plt.legend()
    plt.xlabel('Episode Index')
    plt.ylabel(y_label)
    plt.show()
    fig.savefig(path, format='eps', dpi=1000)


# In[12]:


win = 6
a1,b1,c1,d1,name1 = read_log('t_M_05_nB/max_eps_2000/', user_idx=0, b=0, e=10)
a2,b2,c2,d2,name2 = read_log('t_M_05_nB_dqn/select/', user_idx=0, b=0, e=10)

a11,b11,c11,d11,name11 = read_log('t_M_05_nB/max_eps_2000/', user_idx=1, b=0, e=10)
a12,b12,c12,d12,name12 = read_log('t_M_05_nB_dqn/select/', user_idx=1, b=0, e=10)

a21,b21,c21,d21,name21 = read_log('t_M_05_nB/max_eps_2000/', user_idx=2, b=0, e=10)
a22,b22,c22,d22,name22 = read_log('t_M_05_nB_dqn/select/', user_idx=2, b=0, e=10)

plot_now(a1, a2, a11, a12, a21, a22, 'Average Reward Per Episode', 'figs/t_05_nB_reward.eps')
plot_now(b1, b2, b11, b12, b21, b22, 'Average Power Per Episode', 'figs/t_05_nB_power.eps')
plot_now(c1, c2, c11, c12, c21, c22, 'Average Delay Per Episode', 'figs/t_05_nB_delay.eps')


# In[31]:


a,b,c,d,name = read_log('t_M_08_nB_dqn/', user_idx=0, b=10, e=20)
print (a)
print(np.mean(a), np.mean(b), np.mean(c))
# print(np.mean(b[:,1800:]), np.mean(c[:,1800:]))
# plot_curve(a,b,c,d)
# plot_curve(a[:,500:],b[:,500:],c[:,500:],d[:,500:])


# In[23]:


import numpy as np
path = 't_M_08_nB_dqn/'
u_id = 2
a = [ read_log(path, user_idx=u_id, b=i)[0][0] for i in range(10,20)]
print (np.mean(np.array(a), axis=1))
b = [ read_log(path, user_idx=u_id, b=i)[1][0] for i in range(10,20)]
print (np.mean(np.array(b), axis=1))
c = [ read_log(path, user_idx=u_id, b=i)[2][0] for i in range(10,20)]
print (np.mean(np.array(c), axis=1))


# In[14]:


rate = np.linspace(1.5,4,6)
pg05 = [-3.2195,-4.522,-6.9655,-9.665,-12.459,-15.458]
qn05 = [-3.9055,-5.573,-7.833,-10.4985,-13.2885,-16.3085]
lgd = [-4.502,-6.6195,-8.8395,-11.1235,-13.499,-16.0385]
ogd = [-4.4025,-6.2145,-8.4295,-10.824,-13.173,-15.6555]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')

# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Reward')
plt.show()
fig.savefig('figs/t_05_nB_reward_test.eps', format='eps', dpi=1000)


# In[15]:


rate = np.linspace(1.5,4,6)
pg05 = [0.432,0.578,0.93,1.404,1.924,2.494]
qn05 = [0.542,0.737,1.053,1.494,2.003,2.575]
lgd = [0.75,1.122,1.512,1.91,2.315,2.726]
ogd = [0.73,1.041,1.43,1.85,2.25,2.65]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')

# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Power / Watt')
plt.show()
fig.savefig('figs/t_05_nB_power_test.eps', format='eps', dpi=1000)


# In[16]:


rate = np.linspace(1.5,4,6)
pg05 = [2.119,3.264,4.631,5.29,5.678,5.976]
qn05 = [2.391,3.776,5.136,6.057,6.547,6.867]
lgd = [1.504,2.019,2.559,3.147,3.848,4.817]
ogd = [1.505,2.019,2.559,3.148,3.846,4.811]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')
# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Buffering Delay / ms')
plt.show()
fig.savefig('figs/t_05_nB_delay_test.eps', format='eps', dpi=1000)


# In[17]:


rate = np.linspace(1.5,4,6)
pg05 = [-3.0558,-4.7766,-7.8782,-11.3104,-15.24,-19.1388]
qn05 = [-4.6252,-6.768,-10.0342,-13.5286,-17.3482,-20.876]
lgd = [-6.3008,-9.3798,-12.6078,-15.9094,-19.2896,-22.7714]
ogd = [-6.141,-8.7318,-11.9518,-15.4296,-18.7692,-22.1622]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')

# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Reward')
plt.show()
fig.savefig('figs/t_08_nB_reward_test.eps', format='eps', dpi=1000)


# In[18]:


rate = np.linspace(1.5,4,6)
pg05 = [0.294,0.479,0.796,1.178,1.638,2.099]
qn05 = [0.457,0.646,0.957,1.335,1.757,2.154]
lgd = [0.75,1.122,1.512,1.91,2.315,2.726]
ogd = [0.73,1.041,1.43,1.85,2.25,2.65]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')

# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Power / Watt')
plt.show()
fig.savefig('figs/t_08_nB_power_test.eps', format='eps', dpi=1000)


# In[19]:


rate = np.linspace(1.5,4,6)
pg05 = [3.519,4.723,7.551,9.432,10.68,11.734]
qn05 = [4.846,8,11.891,14.243,16.461,18.22]
lgd = [1.504,2.019,2.559,3.147,3.848,4.817]
ogd = [1.505,2.019,2.559,3.148,3.846,4.811]
fig = plt.figure(figsize=(6, 4.5))
plt.plot(rate, pg05, marker='o', label='DDPG')
plt.plot(rate, qn05, marker='s',label='DQN')
plt.plot(rate, ogd, marker='v', label='GD-Offload')
plt.plot(rate, lgd, marker='^', label='GD-Local')
# marker='o', linestyle='dashed',
# ...      linewidth=2, markersize=12

plt.grid(linestyle=':')
plt.legend()
plt.xlabel('Task Arraival Rate / Mbps')
plt.ylabel('Average Buffering Delay / ms')
plt.show()
fig.savefig('figs/t_08_nB_delay_test.eps', format='eps', dpi=1000)


# In[ ]:





# In[ ]:




