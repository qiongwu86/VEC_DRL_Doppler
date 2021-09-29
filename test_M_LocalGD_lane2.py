#!/usr/bin/env python
# coding: utf-8

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mec_env_var import *
from helper import *
import tensorflow as tf
import ipdb as pdb
import time

rate_list = [2.5,3.5,4,4.5,5,5.5,6]

for rate in rate_list:
    for k in range(1,10):
        
        print('---------' + str(k) + '------------')
        
        MAX_EPISODE = 10
        MAX_EPISODE_LEN = 1000

        NUM_T = 1
        NUM_R = 4
        SIGMA2 = 1e-9

        t_factor = 0.9
        noise_sigma = 0.12

        config = {'state_dim':3, 'action_dim':2};
        train_config = {'minibatch_size':64, 'actor_lr':0.0001, 'tau':0.001, 
                        'critic_lr':0.001, 'gamma':0.99, 'buffer_size':250000, 
                        'random_seed':int(time.clock()*1000%1000), 'noise_sigma':noise_sigma, 'sigma2':SIGMA2}
        
    #     rate = 2.0
        res_path = 'test_M_GD_Local_lane2_rate_'+str(rate)+'/'
        init_path = ''

        #choose the vehicle for testing
        Train_vehicle_ID = 1
        step_result_path = res_path + 'step_result/'
        
        if not os.path.exists(res_path):
            os.mkdir(res_path) 
        if not os.path.exists(step_result_path):
            os.mkdir(step_result_path)
            
        user_config = [{'id':'1', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':1, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':2.0},
                       {'id':'2', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':1, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0},
                       {'id':'3', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':2, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0},
                       {'id':'4', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':2, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':3.0},
                       ]

        
        if not os.path.exists(res_path):
            os.mkdir(res_path) 
            
        print(user_config)

        # 1. include all user in the system according to the user_config
        user_list = [];
        for info in user_config:
            info.update(config)
            user_list.append(MecTermGD(info, train_config, 'local'))
            print('Initialization OK!----> user ' + info['id'])

        # 2. create the simulation env
        env = MecSvrEnv(user_list, Train_vehicle_ID, SIGMA2, MAX_EPISODE_LEN, mode='test')

        res_r = []
        res_p = []
        res_b = []
        res_o = []
        res_d = []
        # 3. start to explore for each episode
        for i in range(MAX_EPISODE):

            cur_init_ds_ep = env.reset()

            step_reward = []
            step_power = []
            step_buffer = []  
            step_power_local = []
            step_power_offload = []  
                 
            cur_r_ep = 0
            cur_p_ep = 0
            cur_op_ep = 0
            cur_ts_ep = 0
            cur_ps_ep = 0 
            cur_rs_ep = 0
            cur_ds_ep = 0
            cur_ch_ep = 0
            cur_of_ep = 0
            cur_power_local = 0
            cur_power_offload = 0
            cur_noise_ep = [0,0]


            for j in range(MAX_EPISODE_LEN):
                # first try to transmit from current state
                [cur_r, done, cur_p, cur_op, temp, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_of, cur_pt, cur_pl, sinr, overdata] = env.step_transmit()
                step_reward.append(cur_r)
                step_power.append(cur_p)
                step_buffer.append(cur_ds)
                step_power_local.append(cur_pl)
                step_power_offload.append(cur_pt)

                cur_r_ep += cur_r
                cur_p_ep += cur_p
                cur_op_ep += cur_op
                cur_ts_ep += cur_ts
                cur_ps_ep += cur_ps
                cur_rs_ep += cur_rs
                cur_ds_ep += cur_ds
                cur_ch_ep += cur_ch
                cur_of_ep += cur_of
                cur_noise_ep += temp
                cur_power_local += cur_pl
                cur_power_offload += cur_pt            
                # print ('%d.....cur_r:%d,cur_p:%d,cur_b:%d'%(j,cur_r,cur_p,cur_ds))

                if done:
                    res_r.append(cur_r_ep/MAX_EPISODE_LEN)
                    res_p.append(cur_p_ep/MAX_EPISODE_LEN)
                    res_b.append(cur_ds_ep/MAX_EPISODE_LEN)
                    res_o.append(cur_of_ep/MAX_EPISODE_LEN)
                    res_d.append(cur_ds)
                    # print('%d:r:%s,p:%s,op:%s,tr:%s,pr:%s,rev:%s,dbuf:%s,ch:%s,ibuf:%s,rbuf:%s' % (i, cur_r_ep/MAX_EPISODE_LEN, cur_p_ep/MAX_EPISODE_LEN, cur_op_ep/MAX_EPISODE_LEN, cur_ts_ep/MAX_EPISODE_LEN, cur_ps_ep/MAX_EPISODE_LEN, cur_rs_ep/MAX_EPISODE_LEN, cur_ds_ep/MAX_EPISODE_LEN, cur_ch_ep/MAX_EPISODE_LEN, cur_init_ds_ep, cur_ds))
                    print('%d:r:%s,p:%s,[%s,%s]dbuf:%s,noise:%s'%(i,cur_r_ep/MAX_EPISODE_LEN,cur_p_ep/MAX_EPISODE_LEN,cur_power_offload/MAX_EPISODE_LEN,cur_power_local/MAX_EPISODE_LEN,cur_ds_ep/MAX_EPISODE_LEN,cur_noise_ep/MAX_EPISODE_LEN))

            step_result = step_result_path+'step_result_'+time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
            np.savez(step_result, step_reward, step_power, step_buffer, step_power_local, step_power_offload)
            time.sleep(1) #pause for a while to save all data 

        name = res_path+'test_1000_' + time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
        np.savez(name, res_r, res_p, res_b, res_o, res_d)