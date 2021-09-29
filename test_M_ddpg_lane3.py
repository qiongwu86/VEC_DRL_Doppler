import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mec_env_var import *
from helper import *
import tensorflow as tf
import tflearn
import ipdb as pdb
import time

rate_list = [2.5,3,3.5,4,4.5,5,5.5,6]

for rate in rate_list:
    
    for k in range(1,10):
        
        tf.compat.v1.reset_default_graph()
        
        print('---------' + str(k) + '------------')
        
        MAX_EPISODE = 10
        MAX_EPISODE_LEN = 1000
      
        NUM_T = 1
        NUM_R = 4
        SIGMA2 = 1e-9
        # slottime (ms)
        slottime = 4

        t_factor = 0.9
        noise_sigma = 0.02

        config = {'state_dim':3, 'action_dim':2};
        train_config = {'minibatch_size':64, 'actor_lr':0.0001, 'tau':0.001, 
                        'critic_lr':0.001, 'gamma':0.998, 'buffer_size':250000, 
                        'random_seed':int(time.perf_counter()*1000%1000), 'noise_sigma':noise_sigma, 'sigma2':SIGMA2}

        IS_TRAIN = False

        res_path = 'test_M_ddpg_sigma0_02_rate' + str(rate) + '_lane2/'
        step_result_path = res_path + 'step_result/'
        model_fold = 'model_M_ddpg_sigma0_02_rate3_lane2/'
        model_path = 'model_M_ddpg_sigma0_02_rate3_lane2/my_train_model_rate_3_' + str(9) + '-2000'

        if not os.path.exists(res_path):
            os.mkdir(res_path) 
        if not os.path.exists(step_result_path):
            os.mkdir(step_result_path) 

        meta_path = model_path + '.meta'
        init_path = ''
        init_seqCnt = 40

        #choose the vehicle for training
        Train_vehicle_ID = 1

        user_config = [{'id':'1', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':1, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':2.0},
                       {'id':'2', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':1, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0},
                       {'id':'3', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':2, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':1.0},
                       {'id':'4', 'model':'AR', 'num_r':NUM_R, 'rate':rate, 'action_bound':2, 
                        'data_buf_size':100, 't_factor':t_factor, 'penalty':1000,'lane':3.0},
                       ]
        print(user_config)

        # 0. initialize the session object
        sess = tf.compat.v1.Session() 

        # 1. include all user in the system according to the user_config
        user_list = [];
        for info in user_config:
            info.update(config)
            info['model_path'] = model_path 
            info['meta_path'] = info['model_path']+'.meta'
            info['init_path'] = init_path
            info['init_seqCnt'] = init_seqCnt
            user_list.append(MecTermLD(sess, info, train_config))
            print('Initialization OK!----> user ' + info['id'])

        # 2. create the simulation env
        env = MecSvrEnv(user_list, Train_vehicle_ID, SIGMA2, MAX_EPISODE_LEN, mode='test')

        res_r = []
        res_p = []
        res_p_offload = []
        res_p_local=[]
        res_b = []
        res_o = []
        res_d = []

        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # 3. start to explore for each episode
        for i in range(MAX_EPISODE):
            # plt.ion()
            cur_init_ds_ep = env.reset()

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
            step_pl = []
            step_po = []
            step_sinr = []
            step_buffer = []
            step_local_bit = []
            step_offload_bit = []
            step_overdata = []
            step_sumdata = []
            step_reward = []
            step_power = []

            for j in range(MAX_EPISODE_LEN):
                # first try to transmit from current state
                [cur_r, done, cur_p, cur_op, temp, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_of, cur_pt, cur_pl, sinr, overdata] = env.step_transmit()
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

                step_pl.append(cur_pl)
                step_po.append(cur_pt)
                step_reward.append(cur_r)
                step_power.append(cur_p)
                step_buffer.append(cur_ds)

                if done:
                    res_r.append(cur_r_ep/MAX_EPISODE_LEN)
                    res_p.append(cur_p_ep/MAX_EPISODE_LEN)
                    res_b.append(cur_ds_ep/MAX_EPISODE_LEN)
                    res_o.append(cur_of_ep/MAX_EPISODE_LEN)
                    res_p_local.append(cur_power_local/MAX_EPISODE_LEN)
                    res_p_offload.append(cur_power_offload/MAX_EPISODE_LEN)
                    res_d.append(cur_ds)
                    # plt.pause(0.000000000001)

                    # try:
                    #     ax.lines.remove(line_localpower[0])
                    #     ax.lines.remove(line_offloadpower[0])
                    # except Exception:
                    #     pass
                    
                    # line_localpower = ax.plot(range(0,MAX_EPISODE_LEN),step_pl,'#ff7f0e',label='localower',lw=0.8)
                    # line_offloadpower= ax.plot(range(0,MAX_EPISODE_LEN),step_po,'#1f77b4',label='offloadpower',lw=0.8)
                    # ax.legend()
                    print('%d:r:%s,p:%s,[%s,%s]dbuf:%s,noise:%s'%(i,cur_r_ep/MAX_EPISODE_LEN,cur_p_ep/MAX_EPISODE_LEN,cur_power_offload/MAX_EPISODE_LEN,cur_power_local/MAX_EPISODE_LEN,cur_ds_ep/MAX_EPISODE_LEN,cur_noise_ep/MAX_EPISODE_LEN))

            step_result = step_result_path+'step_result_'+time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
            np.savez(step_result, step_reward, step_power, step_buffer, step_pl, step_po)
            time.sleep(0.8) #pause for a while to save all data 

        # plt.ioff()

        name = res_path+'test' + time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
        np.savez(name, res_r, res_p, res_b, res_o, res_d, res_p_local, res_p_offload)

        sess.close()


