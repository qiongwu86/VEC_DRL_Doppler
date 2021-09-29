
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
# dis_init=np.floor(np.random.uniform(-50,50,3))
# print(dis_init[0],dis_init[1],dis_init[2])
# np.random.seed([1])
# def complexGaussian(row=1, col=1, amp=1.0):
#     real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
#     img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
#     return amp*(real + 1j*img)
# print(np.random.normal(size=[1,4]))
# np.random.seed([1])

# print(np.random.normal(size=[1,4])[0])

# a = np.random.normal(size=[1,4])
# print(a[0][0])

# powers =[[1,2],[3,4],[5,6]]
# powers = np.array(powers)
# print(powers[:,0])


# dis_init=np.floor(np.random.uniform(-50,50))
# print (dis_init)

# print(int(time.perf_counter()*1000%1000))
# print(np.random.normal(size=[row,col]))

# x = np.array([1,2,3])
# print (np.power(np.linalg.norm(x),2))

# class add_1(object):
# 	"""docstring for add_1"""
# 	def __init__(self, arg):
# 		# super(add_1, self).__init__()
# 		self.arg = arg

# 	def add(self):
# 		self.arg += 1
# 		return self.arg
# 	def getnum(self):
# 		return self.arg	

# lis= [1,2,3,4]

# lis_2 = []
# for x in lis:
# 	lis_2.append(add_1(x))
# # print ()

# b = np.array([n.add() for n in lis_2 if n.getnum()>=2 and n.getnum()<=3])
# print (b)

# a = -49
# b=40
# print ((a+50)//100==(b+50)//100)

# print('%s:%d'%('ab',3),45)

# for x in range(1,10):
# 	if x != 2:
# 		print(x)
# print(np.random.randint(-50,51))

# import tensorflow as tf
# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

# table = np.array([[float(2)/(8-1)*i for i in range(8)] for j in range(2)])
# print  (table)

# print (1<2 or 2<3)
# print (np.random.randint(0, 25))

# print (np.power(0.999976,200000))
# power = 0.28054524015134374
# databuffer =   40.0
# print (power*5 + databuffer*0.5)





# def add_layer(inputs,in_size,out_size,activation_funiction=None):
 
#     Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#     biases = tf.Variable(tf.zeros([1,out_size]) +0.1)
#     Wx_plus_b = tf.matmul(inputs,Weights)+biases
#     if activation_funiction is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_funiction(Wx_plus_b)
#     return outputs
 
# x_data = np.linspace(-1,1,300)[:,np.newaxis]
# noise = np.random.normal(0,0.05,x_data.shape)
# y_data = np.square(x_data)-0.5 +noise
 
# xs = tf.placeholder(tf.float32,[None,1])   
# ys = tf.placeholder(tf.float32,[None,1])
 
# #add hidden layer
# l1 = add_layer(xs,1,10,activation_funiction=tf.nn.relu)
# #add output layer
# prediction = add_layer(l1,10,1,activation_funiction=None)
 
# #the error between prediction and real data
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
# init =tf.initialize_all_variables()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# # ax2 = fig.add_subplot(1,2,2)

# ax.scatter(x_data,y_data)
# with tf.Session() as sess:
#     sess.run(init)
 

#     # plt.ion()   #将画图模式改为交互模式
 
#     for i in range(1000):
#         sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#         plt.ion()
#         if i%50 ==0:
#         	plt.pause(0.3)
#         	try:
# 	        	ax.lines.remove(lines[0])
# 	        	ax.lines.remove(lines2[0])
#         	except Exception:
#         		pass
#         	prediction_value = sess.run(prediction,feed_dict={xs:x_data})
#         	lines = ax.plot(x_data,prediction_value,'r-',lw=5)
#         	lines2 = ax.plot(x_data,prediction_value+1,'y-',lw=5)

#         	print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
 
#     plt.ioff()
#     # plt.show()
# print (np.power(300.,4))
# print (np.power(90149./149.,2)*0.95)

# a = np.array([ 2.62496285e-05-1.89765586e-05j,  3.56032445e-05-1.11978915e-05j,
#        -3.82152427e-05+2.13053911e-05j,  1.90245113e-05+2.51402042e-05j])
# print(np.linalg.norm(a, axis=0))
# print(np.log2(a, axis=0))

# k = 1e-28
# slottime = 0.02 #unit: s
# bandwith = 1 #unit: MHz
# L = 500 


# fig = plt.figure()
# power_range = np.linspace(0,1,10000)[:,np.newaxis]
# plt.grid(linestyle=':')


# for sinr in range(20,40000,10):
# 	offload_ability = np.log2(1 + power_range*sinr)*slottime*bandwith*1000
# 	plt.plot(power_range, offload_ability, color='cornflowerblue', lw= 0.5 )
# plt.plot(power_range, np.log2(1 + power_range*40000)*slottime*bandwith*1000, label='sinr40000',color='#1f77b4', lw= 1 )
# plt.plot(power_range, np.log2(1 + power_range*100)*slottime*bandwith*1000, label='sinr100',color='#1f77b4', lw= 1 )
# plt.plot(power_range, np.log2(1 + power_range*300)*slottime*bandwith*1000, label='sinr300',color='#1f77b4', lw= 1 )
# plt.plot(power_range, np.log2(1 + power_range*1000)*slottime*bandwith*1000, label='sinr1000',color='#1f77b4', lw= 1 )
# plt.plot(power_range, np.log2(1 + power_range*5000)*slottime*bandwith*1000, label='sinr5000',color='#1f77b4', lw= 1 )

# localpocess_ability = np.power(power_range/k, 1.0/3.0)*slottime/L/1000
# plt.plot(power_range, localpocess_ability, color='#ff7f0e', label='localpocess_ability',lw= 2 )

# plt.ylabel('process ability')
# plt.xlabel('power')

# plt.show()

# print (np.power(0.999976,100000),np.power(0.1,1/500000),np.power(0.99996,100000))



