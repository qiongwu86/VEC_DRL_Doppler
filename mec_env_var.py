import numpy as np
from helper import *
import ipdb as pdb
import tensorflow as tf
import time
from scipy import special as sp
from scipy.constants import pi


class MecTerm(object):
    """
    MEC terminal parent class
    """
    def __init__(self, user_config, train_config):
        self.rate = user_config['rate']
        self.dis = 0
        self.lane = user_config['lane']
        self.id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.data_buf_size = user_config['data_buf_size']
        self.t_factor = user_config['t_factor']
        self.penalty = user_config['penalty']
        self.seed = train_config['random_seed']
        self.sigma2 = train_config['sigma2']
        self.lamda = 7
        self.train_config = train_config

        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        self.n_t = 1
        self.n_r = user_config['num_r']    
        self.DataBuf = 0
        
        self.SINR = 0
        self.Power = np.zeros(self.action_dim)
        self.Reward = 0
        self.State = []
        
        # some pre-defined parameters
        self.k = 1e-29
        self.t = 0.02
        self.L = 500 
        self.bandwidth = 1 #MHz
        self.velocity_lane1 = 20.0
        self.velocity_lane2 = 25.0
        self.velocity_lane3 = 30.0
        # self.channelModel = ARModel(self.n_t, self.n_r, rho=compute_rho(self) ,seed=train_config['random_seed'])
        self.channelModel = ARModel(self.n_t, self.n_r, seed=self.train_config['random_seed'])

    def dis_mov(self):
        if self.lane == 1:
            self.dis += (self.velocity_lane1*self.t)
        elif self.lane == 2:
            self.dis += (self.velocity_lane2*self.t)
        elif self.lane == 3:
            self.dis += (self.velocity_lane3*self.t)
        return self.dis

    def compute_rho(self):
        width_lane = 5
        Hight_RSU = 10
        x_0 = np.array([1,0,0])
        P_B = np.array([0,0,Hight_RSU])
        P_m = np.array([self.dis, width_lane*self.lane, Hight_RSU])
        if self.lane == 1:
            self.rho  = sp.j0(2*pi*self.t*self.velocity_lane1*np.dot(x_0,(P_B-P_m))/(np.linalg.norm(P_B-P_m)*self.lamda))
        elif self.lane == 2:
            self.rho  = sp.j0(2*pi*self.t*self.velocity_lane2*np.dot(x_0,(P_B-P_m))/(np.linalg.norm(P_B-P_m)*self.lamda))
        elif self.lane == 3:
            self.rho  = sp.j0(2*pi*self.t*self.velocity_lane3*np.dot(x_0,(P_B-P_m))/(np.linalg.norm(P_B-P_m)*self.lamda))
        return self.rho

    def sampleCh(self):
        self.compute_rho()
        self.Channel = self.channelModel.sampleCh(self.dis,self.rho,self.lane)
        return self.Channel

    def getCh(self):
        self.Channel = self.channelModel.getCh(self.dis,self.lane)    
        return self.Channel
   
    def setSINR(self, sinr):
        self.SINR = sinr
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.State = np.array([self.DataBuf, sinr, self.dis])

    def localProc(self, p):
        return np.power(p/self.k, 1.0/3.0)*self.t/self.L/1000 #unit:kbit
    
    def localProcRev(self, b):
        return np.power(b*1000*self.L/self.t, 3.0)*self.k
    
    def offloadRev(self, b):
        return (np.power(2.0, b/(self.t*self.bandwidth*1000))-1)*self.sigma2/np.power(np.linalg.norm(self.Channel),2)
    
    def offloadRev2(self, b):
        return self.action_bound if self.SINR <= 1e-12 else (np.power(2.0, b)-1)/self.SINR
        
    def sampleData(self):
        data_t = np.log2(1 + self.Power[0]*self.SINR)*self.t*self.bandwidth*1000 #unit:kbits
        data_p = self.localProc(self.Power[1])
        over_power = 0
        self.DataBuf -= data_t+data_p
        
        if self.DataBuf < 0:
            over_power = self.Power[1] - self.localProcRev(np.fmax(0, self.DataBuf+data_p))
            self.overdata = -self.DataBuf
            self.DataBuf = 0
        else:
            self.overdata = 0
        data_r = np.random.poisson(self.rate) #unit :mbit
        # data_r = 4  #unit：mbps
        # print('data_r:',data_r)
        self.DataBuf += data_r*self.t*1000 #unit:kbit
        # print (data_t,data_p)
        return data_t, data_p, data_r, over_power, self.overdata
    
    def buffer_reset(self, rate, seqCount):
        self.rate = rate
        self.DataBuf = np.random.randint(0, self.data_buf_size-1)/2.0
        self.sampleCh()
        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True
        return self.DataBuf

    def dis_reset(self):
        if int(self.id) == 1 :
            self.dis = -250
        elif int(self.id)==2 :
            self.dis = -450 
        elif int(self.id)==3 :
            self.dis = -500
        elif int(self.id)==4 :
            self.dis = -550

    def disreset_for_test(self):
        if int(self.id) == 1 :
            self.dis = -250
        elif int(self.id)==2 :
            self.dis = -450 
        elif int(self.id)==3 :
            self.dis = -500
        elif int(self.id)==4 :
            self.dis = -550


class MecTermLD(MecTerm):
    """
    MEC terminal class for loading from stored models
    """
    
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        
        saver = tf.train.import_meta_graph(user_config['meta_path'])
        saver.restore(sess, user_config['model_path'])
 
        graph = tf.get_default_graph()
        input_str = "input_" + self.id + "/X:0"
        output_str = "output_" + self.id + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)

    def feedback(self, sinr):
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        # estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        
        # update system state
        self.State = self.next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata
    
    def predict(self, isRandom):
        self.Power = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        return self.Power, np.zeros(self.action_dim)
        
class MecTermDQN_LD(MecTermLD):
    """
    MEC terminal class for loading from stored models of DQN
    """
    def __init__(self, sess, user_config, train_config):
        MecTermLD.__init__(self, sess, user_config, train_config)
        graph = tf.get_default_graph()
        self.action_level = user_config['action_level']
        self.action = 0
        
        output_str = "output_" + self.id + "/BiasAdd:0"
        self.out = graph.get_tensor_by_name(output_str)
        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])
        
    def predict(self, isRandom):
        q_out = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        self.action = np.argmax(q_out)
        action_tmp = self.action
        for i in range(self.action_dim):
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        return self.Power, np.zeros(self.action_dim)
        
class MecTermGD(MecTerm):
    """
    MEC terminal class using Greedy algorithms
    """
    
    def __init__(self, user_config, train_config, policy):
        MecTerm.__init__(self, user_config, train_config)
        self.policy = policy #         
        self.local_proc_max_bits = self.localProc(self.action_bound) # max processed bits per slot
        
    def feedback(self, sinr):
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []

        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()
        
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf 

        # if self.DataBuf > self.data_buf_size:
            # isOverflow = 1
            # self.DataBuf = self.data_buf_size
        self.dis_mov()
        self.sampleCh()
        
        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])

        # update system state
        self.State = self.next_state
        # print (self.Power)
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        # return self.Reward, np.sum(self.Power), 0, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata

    def predict(self, isRandom):
        data = self.DataBuf
        if self.policy == 'local':
            self.offloadDo(self.localProcDo(data))
        else: 
            self.localProcDo(self.offloadDo(data))
        
        self.Power = np.fmax(0, np.fmin(self.action_bound, self.Power))
        # print ('power:',self.Power)
        return self.Power, np.zeros([self.action_dim])
    
    def localProcDo(self, data):
        if self.local_proc_max_bits <= data:
            self.Power[1] = self.action_bound
            data -= self.local_proc_max_bits
        else:
            self.Power[1] = self.localProcRev(data)
            data = 0
        return data
    
    def offloadDo(self, data):
        offload_max_bits = np.log2(1 + self.action_bound*self.SINR)*self.t*self.bandwidth*1000
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev(data)
            data = 0
        return data
    
class MecTermGD_M(MecTermGD):
    def offloadDo(self, data):
        offload_max_bits = np.log2(1+self.SINR*self.action_bound)
        if offload_max_bits <= data:
            self.Power[0] = self.action_bound
            data -= offload_max_bits
        else:
            self.Power[0] = self.offloadRev2(data)
            data = 0
        return data

class MecTermRL(MecTerm):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, user_config, train_config)
        
        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False

    def feedback(self, sinr):
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf 
        # self.Reward = -self.t_factor*(self.Power[1])*10 - self.Power[0]*np.log2(sinr)- (1-self.t_factor)*self.DataBuf - (1-self.t_factor)*overdata

        # estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        
        # update system state
        
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        # return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata


    def predict(self, isRandom):
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        self.Power = np.fmax(0, np.fmin(self.action_bound, power))
        # self.Power = [0.01,0.1]
        return self.Power, noise

    def AgentUpdate(self,done):
        self.agent.update(self.State, self.Power, self.Reward, done, self.next_state, self.isUpdateActor)
        self.State = self.next_state
       
class MecTermDQN(MecTerm):
    """
    MEC terminal class using DQN
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.action_level = user_config['action_level']
        self.agent = DQNAgent(sess, user_config, train_config)
        self.action = 0

        self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])


    def feedback(self, sinr):
        isOverflow = 0
        self.SINR = sinr
        self.next_state = []
        # update the data buffer
        [data_t, data_p, data_r, over_power, overdata] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor*np.sum(self.Power)*10 - (1-self.t_factor)*self.DataBuf

        # estimate the channel for next slot
        self.dis_mov()
        self.sampleCh()

        # update the channel and state
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        self.next_state = np.array([self.DataBuf, sinr, self.dis])
        # update system state
        
        # return the reward in this slot
        sum_power = np.sum(self.Power)
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow, self.Power[0],self.Power[1], overdata


    def AgentUpdate(self,done):
        self.agent.update(self.State, self.action, self.Reward, done, self.next_state)
        self.State = self.next_state
       
    def predict(self, isRandom):
        # print ('self.table:',self.table)
        self.action, noise = self.agent.predict(self.State)
        # print ('action:',self.action)
        action_tmp = self.action
        for i in range(self.action_dim):
            # print ('action_tmp ,self.action_level',action_tmp,self.action_level)
            self.Power[i] = self.table[i, action_tmp % self.action_level]
            action_tmp //= self.action_level
        # print ( 'self.Power:',self.Power)
        return self.Power, noise

class MecSvrEnv(object):
    """
    Simulation environment
    """
    def __init__(self, user_list, Train_vehicle_ID, sigma2, max_len,mode='train'): 
        self.user_list = user_list
        self.num_user = len(user_list)
        self.Train_vehicle_ID = Train_vehicle_ID-1
        self.sigma2 = sigma2
        self.count = 0
        self.seqCount = 0
        self.max_len = max_len
        self.mode = mode
        # self.seed = 0

    def init_target_network(self):
        self.user_list[self.Train_vehicle_ID].agent.init_target_network()

    def step_transmit(self, isRandom=True):
        #the id of vehicle for training
        i = self.Train_vehicle_ID 
        # get the channel vectors 
        # print('\ncount:',self.count)
        # print('\ndis of step %d:'%self.count,[user.dis for user in self.user_list])
        
        channels = []
        channels.append(self.user_list[i].getCh())
        for user in self.user_list:
            #ensure whether two user are in the same BS' cover area 
            if (user.dis+250)//500 == (self.user_list[i].dis+250)//500 and int(user.id)!=self.Train_vehicle_ID+1:
                channels.append(user.getCh())

        # channels = np.transpose(channels)
        # channels = np.transpose([user.getCh() for user in self.user_list if (user.dis+50)//100 == (self.user_list[i].dis+50)//100])
        # print('channels:',np.linalg.norm(channels, axis=1))
        # get the transmit powers 
        powers, noises = self.user_list[i].predict(isRandom)

        sinr_list = self.compute_sinr(channels)

        rewards = 0
        powers = 0
        over_powers = 0
        data_ts = 0
        data_ps = 0
        data_rs = 0
        data_buf_sizes = 0
        next_channels = 0
        isOverflows = 0
        power_offload = 0
        power_local = 0
        
        self.count += 1

        # print('sinr_list:',sinr_list)
        # feedback the sinr to each user
        [rewards, powers, over_powers, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows, power_offload, power_local ,overdata] = self.user_list[i].feedback(sinr_list[0])
        # print ('self.mode:',self.mode)
        if self.mode == 'train':
            self.user_list[i].AgentUpdate(self.count >= self.max_len)

        # update the distance of other vehicle that dont training 
        for user in self.user_list:
            if int(user.id) != self.Train_vehicle_ID + 1:
                user.dis_mov()

        return rewards, self.count >= self.max_len, powers, over_powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows, power_offload, power_local, sinr_list[0],overdata


    def compute_sinr(self, channels):
        # Spatial-Domain MU-MIMO ZF
        H_inv = np.linalg.pinv(np.transpose(channels))
        noise = np.power(np.linalg.norm(H_inv, axis=1),2)*self.sigma2
        sinr_list = 1/noise
        return sinr_list

    def reset(self, isTrain=True):
        #the id of vehicle for training
        i = self.Train_vehicle_ID
        self.count = 0
        if isTrain:
            init_data_buf_size = self.user_list[i].buffer_reset(self.user_list[i].rate, self.seqCount) 
            # print('initial buffer:',self.user_list[i].DataBuf)
            # dis = [user.dis_reset() for user in self.user_list]
            for user in self.user_list:
                if self.mode == 'train':
                    user.dis_reset()
                elif self.mode == 'test':
                    user.disreset_for_test()

            # get the channel vectors   
            channels = [user.getCh() for user in self.user_list]
            # print (channels,'\n',np.linalg.norm(channels, axis=1))
            # compute the sinr for each user
            sinr_list = self.compute_sinr(channels)
            # print (sinr_list)

        else:
            init_data_buf_size = 0 
            sinr_list = [0 for user in self.user_list]

        self.user_list[i].setSINR(sinr_list[i])
            
        self.seqCount += 1
        return init_data_buf_size


