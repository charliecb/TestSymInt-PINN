import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
from sympy import *
from SPINN import funcODE_SPINN

'''
Define the hyper-parameter
============================================
'''
np.random.seed(1234)
tf.random.set_seed(1243)

N_f = 401
N_t = 10
lyscl = [10,10]

data_type = tf.float64

'''
Define the problem
============================================
'''
x_train = tf.linspace(-1,1,N_t)[:,None]#, dtype=data_type)#tf.constant([0.,1.,2.], dtype=data_type)[:,None]
u_train = tf.exp(x_train)
x_star = np.linspace(-1,1,N_f)[:,None]
u_star = np.exp(x_star)

# Doman bounds
lx = x_star.min()
ux = x_star.max()

'''
Conducting the first-round prediction
============================================
'''
# x_f_train = lx + (ux-lx)*lhs(1, 200)
x_f_train = np.linspace(-1,1,5)[:,None]
#x_f_train = lx + (ux-lx)*lhs(1, N_t)

#x_f_train = np.vstack((x_f_train, x_train))
x_f_train = tf.cast(x_f_train, dtype=data_type)

gamma = 1/3
width = 10

gov_eqn = lambda expression, variable: diff(expression, variable) - (expression)
    
model = funcODE_SPINN(x_train, u_train, gov_eqn, width, lyscl, lx, ux, gamma)

start_time = time.time() 
model.train(500)
elapsed = time.time() - start_time                
print('Training time: %.4f' % (elapsed))

x_star = tf.cast(x_star, dtype=data_type)
u_pred = model.predict(x_star)


######################################################################
############################# Plotting ###############################
######################################################################    

fig = plt.figure(figsize = [10, 10], dpi = 300)

ax = plt.subplot(211)
ax.plot(x_star, u_star, 'b-', linewidth = 2, label = 'exact')
ax.plot(x_star, u_pred, 'r--', linewidth = 2, label = 'predict')
ax.scatter(x_train, u_train, c='g', label = 'datapoints')

ax.set_xlabel('$x$', fontsize = 15)
ax.set_ylabel('$u$', fontsize = 15, rotation = 0)
ax.set_title('solution', fontsize = 10)
