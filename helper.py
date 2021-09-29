import os
import numpy as np
import tensorflow as tf
from ddpg_lib import * 
import ipdb as pdb
import matplotlib.pyplot as plt

alpha = 2.0
ref_loss = 0.001
width_lane = 5
Hight_RSU = 10 

def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    return amp*(real + 1j*img)

class ARModel(object):
    """docstring for AR channel Model"""
    def __init__(self, n_t=1, n_r=1, seed=123):
        self.n_t = n_t
        self.n_r = n_r
        np.random.seed([seed])       
        self.H = complexGaussian(self.n_t, self.n_r)  
        
    def getCh(self,dis,lane):
        dis_lane = width_lane*lane
        self.dis = dis
        self.position = np.array([self.dis,dis_lane,Hight_RSU])
        self.path_loss = ref_loss/np.power(np.linalg.norm(self.position),2)
        return self.H*np.sqrt(self.path_loss)
        
    def sampleCh(self,dis,rho,lane):
        self.H = rho*self.H + complexGaussian(self.n_t, self.n_r, np.sqrt(1-rho*rho))
        return self.getCh(dis,lane)

class DQNAgent(object):
    """docstring for DQNAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.action_level = user_config['action_level']
        self.minibatch_size = int(train_config['minibatch_size'])
        self.epsilon = float(train_config['epsilon'])
        
        self.action_nums = 1
        for i in range(self.action_dim):
            self.action_nums *= self.action_level
        
        self.max_step = 500000
        self.pre_train_steps = 25000
        self.total_step = 0
        self.DQN = DeepQNetwork(sess, self.state_dim, self.action_nums, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.user_id)
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

    def init_target_network(self):
        self.DQN.update_target_network()
        
    def predict(self, s):
        if self.total_step <= self.max_step:
            self.epsilon *= 0.9999953948404178
        # print (self.epsilon)
        # print (np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps)
        if np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps:
            action = np.random.randint(0, self.action_nums)
        else:
            action, _ = self.DQN.predict(np.reshape(s, (1, self.state_dim)))
        self.total_step += 1
        # print ('self.total_step:',self.total_step)
        # print (' self.epsilon:', self.epsilon)
        return action, np.zeros([1])
    
    def update(self, s, a, r, t, s2):
        self.replay_buffer.add(np.reshape(s, (self.state_dim,)), a, r,
                              t, np.reshape(s2, (self.state_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            _, q_out = self.DQN.predict(s_batch)
            target_prediction, target_q_out = self.DQN.predict_target(s2_batch)
            
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    q_out[k][a_batch[k]] = r_batch[k]
                else:
                    q_out[k][a_batch[k]] = r_batch[k] + self.DQN.gamma * target_q_out[k][target_prediction[k]]

            # Update the critic given the targets
            q_loss, _ = self.DQN.train(
                s_batch, q_out) 
            
            # losses.append(q_loss)
            # Update target networks
            self.DQN.update_target_network()

class DDPGAgent(object):
    """docstring for DDPGAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.init_path = user_config['init_path'] if 'init_path' in user_config else ''
        
        self.minibatch_size = int(train_config['minibatch_size'])
        self.noise_sigma = float(train_config['noise_sigma'])

        # initalize the required modules: actor, critic and replaybuffer
        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.action_bound, float(train_config['actor_lr']), float(train_config['tau']), self.minibatch_size, self.user_id)
        
        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.actor.get_num_trainable_vars())
        
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

        # mu, sigma=0.12, theta=.15, dt=1e-2,
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim),sigma=self.noise_sigma) 
        # self.actor_noise = GaussianNoise(0.1, 0.01, size=np.array([self.action_dim]))
        
    def init_target_network(self):
        # Initialize the original network and target network with pre-trained model
        if len(self.init_path) == 0:
            self.actor.update_target_network()
        else:
            self.actor.init_target_network(self.init_path)
        self.critic.update_target_network()

    # input current state and then return the next action
    def predict(self, s, isUpdateActor):
        if isUpdateActor:
            noise = self.actor_noise()
        else:
            noise = np.zeros(self.action_dim)
            
        return self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))[0] + noise, noise
        # return self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + np.random.normal(0.0,0.1,[self.action_dim])

    def update(self, s, a, r, t, s2, isUpdateActor):
        self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
                              t, np.reshape(s2, (self.actor.s_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.minibatch_size, 1)))

            if isUpdateActor:
                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
                      
def test_helper(env, num_steps):
    cur_init_ds_ep = env.reset()
    
    user_list = env.user_list
    cur_r_ep = np.zeros(len(user_list))
    cur_p_ep = np.zeros(len(user_list))
    cur_ts_ep = np.zeros(len(user_list))
    cur_ps_ep = np.zeros(len(user_list))
    cur_rs_ep = np.zeros(len(user_list))
    cur_ds_ep = np.zeros(len(user_list))
    cur_ch_ep = np.zeros(len(user_list))
    
    for j in range(num_steps):
        # first try to transmit from current state
        [cur_r, done, cur_p, cur_n, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch] = env.step_transmit()
        
        cur_r_ep += cur_r
        cur_p_ep += cur_p
        cur_ts_ep += cur_ts
        cur_rs_ep += cur_rs
        cur_ds_ep += cur_ds
        cur_ch_ep += cur_ch        
        
        if cur_r <= -1000:
            print("<-----!!!----->")
        
        print('%d:r:%f,p:%s,n:%s,tr:%s,ps:%s, rev:%s,dbuf:%s,ch:%s,ibuf:%s' % (j, cur_r, cur_p, cur_n, cur_ts, cur_ps, cur_rs, cur_ds, cur_ch, cur_init_ds_ep))
        
    print('r:%.4f,p:%.4f,tr:%.4f,pr:%.4f,rev:%.4f,dbuf:%.4f,ch:%.8f,ibuf:%d' % (cur_r_ep/MAX_EPISODE_LEN, cur_p_ep/MAX_EPISODE_LEN, cur_ts_ep/MAX_EPISODE_LEN, cur_ps_ep/MAX_EPISODE_LEN, cur_rs_ep/MAX_EPISODE_LEN, cur_ds_ep/MAX_EPISODE_LEN, cur_ch_ep/MAX_EPISODE_LEN, cur_init_ds_ep[0]))
        
def plot_everything(res, win=10):
    length = len(res)
    temp = np.array(res)
    
    rewards = temp[:,:,0]
    avg_r = np.sum(rewards, axis=1)/rewards.shape[1]
    plt.plot(range(avg_r.shape[0]), avg_r)
    
    avg_r_sm = moving_average(avg_r, win)
    plt.plot(range(avg_r_sm.shape[0]), avg_r_sm)
    
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
    
    powers = temp[:,:,2]
    avg_p = np.sum(powers, axis=1)/powers.shape[1]
    plt.plot(range(avg_p.shape[0]), avg_p)
    
    avg_p_sm = moving_average(avg_p, win)
    plt.plot(range(avg_p_sm.shape[0]), avg_p_sm)
    
    plt.xlabel('step')
    plt.ylabel('power')
    plt.show()
    
    bufs = temp[:,:,7]
    avg_b = np.sum(bufs, axis=1)/bufs.shape[1]
    plt.plot(range(avg_b.shape[0]), avg_b)
    
    avg_b_sm = moving_average(avg_b, win)
    plt.plot(range(avg_b_sm.shape[0]), avg_b_sm)
    
    plt.xlabel('step')
    plt.ylabel('buffer length')
    plt.show()
    
    ofs = temp[:,:,9]
    avg_o = np.sum(ofs, axis=1)/ofs.shape[1]
    plt.plot(range(avg_o.shape[0]), avg_o)
    
    avg_o_sm = moving_average(avg_o, win)
    plt.plot(range(avg_o_sm.shape[0]), avg_o_sm)
    
    plt.xlabel('step')
    plt.ylabel('buffer length')
    plt.show()
    
    return avg_r, avg_p, avg_b, avg_o

def read_log(dir_path, user_idx=0):
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    avg_ps = []
    avg_bs = []
    avg_os = []

    for name in fileList:
        path = dir_path + name
        res = np.load(path)

        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs[:, user_idx])
        
        temp_ps = np.array(res['arr_1'])
        avg_ps.append(temp_ps[:, user_idx])
        
        temp_bs = np.array(res['arr_2'])
        avg_bs.append(temp_bs[:, user_idx])
        
        temp_os = np.array(res['arr_3'])
        avg_os.append(temp_os[:, user_idx])
    
    avg_rs = np.array(avg_rs)
    avg_ps = np.array(avg_ps)
    avg_bs = np.array(avg_bs)
    avg_os = np.array(avg_os)
    
    return avg_rs, avg_ps, avg_bs, avg_os
    
def plot_curve(rs, ps, bs, os, win=10):
    for avg_r in rs:
        avg_r_sm = moving_average(avg_r, win)
        plt.plot(range(avg_r.shape[0]), avg_r)
        plt.plot(range(avg_r_sm.shape[0]), avg_r_sm)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
    plt.show()
    
    for avg_p in ps:
        avg_p_sm = moving_average(avg_p, win)
        plt.plot(range(avg_p.shape[0]), avg_p)
        plt.plot(range(avg_p_sm.shape[0]), avg_p_sm)
        plt.xlabel('step')
        plt.ylabel('power')
    plt.show()
    
    for avg_b in bs:
        avg_b_sm = moving_average(avg_b, win) 
        plt.plot(range(avg_b.shape[0]), avg_b)
        plt.plot(range(avg_b_sm.shape[0]), avg_b_sm)
        plt.xlabel('step')
        plt.ylabel('buffer length')
    plt.show()
    
    for avg_o in os:
        avg_o_sm = moving_average(avg_o, win) 
        plt.plot(range(avg_o.shape[0]), avg_o)
        plt.plot(range(avg_o_sm.shape[0]), avg_o_sm)
        plt.xlabel('step')
        plt.ylabel('overflow probability')
    plt.show()
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

