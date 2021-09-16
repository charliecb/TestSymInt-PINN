import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import time
from sympy import *
from typing import Callable, List
import SymMath

class funcODE_SPINN:
    
    def __init__(self, x_u, u, gov_eqn, width, lyscl, lb, ub, gamma):
        self.x_u = x_u
        self.u = u
        
        self.lb = lb
        self.ub = ub
        
        self.width = width
        self.lyscl = lyscl
        self.lw = gamma
        self.data_type = x_u.dtype
        
        self.gov_eqn = gov_eqn
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(width, lyscl)
        
        self.parameters = self.weights + self.biases
        
        self.optimizer_Adam = tf.optimizers.Adam()
        self.initialize_sympy()
        
        
    '''
    Sympy specific functionality
    ================================================
    '''
        
    def initialize_sympy(self):
        print("Initializing sympy with governing equation")
        
        x = Symbol('x')
        pprint(self.gov_eqn(Function('u')(x), x))
        
        W = MatrixSymbol('W', 1, self.width)
        M = MatrixSymbol('M', self.width, 1)
        b = MatrixSymbol('b', 1, self.width)
        c = MatrixSymbol('c', 1, 1)
        H = 2.0 * (x - self.lb)/(self.ub-self.lb) - 1
        U_pred = (((W*H+b).applyfunc(sin))*M+c)[0]
        
        print("Computing integral equation loss symbolically. This may take a while...")
        start_time = time.time()
        eqn_loss_sympy = self.compute_integral_loss(U_pred, x)
        self.sympyloss = eqn_loss_sympy
        elapsed = time.time() - start_time
        print("Done constructing loss function! (%.2f seconds)" % elapsed)
        
        print("Now computing symbolic gradients and converting to tensorflow. This should take roughly the same amount of time...")
        start_time = time.time()
        self.eqn_loss = lambdify([W, M, b, c], eqn_loss_sympy, 'tensorflow')
        self.grad_W_eqn = lambdify([W, M, b, c], SymMath.gradient(eqn_loss_sympy, W), 'tensorflow')
        self.grad_b_eqn = lambdify([W, M, b, c], SymMath.gradient(eqn_loss_sympy, b), 'tensorflow')
        self.grad_M_eqn = lambdify([W, M, b, c], SymMath.gradient(eqn_loss_sympy, M), 'tensorflow')
        self.grad_c_eqn = lambdify([W, M, b, c], SymMath.gradient(eqn_loss_sympy, c), 'tensorflow')
        elapsed = time.time() - start_time
        print("Done initializing sympy functionality! (%.2f seconds)" % elapsed)
        print("You can now save these gradient functions for future use.")
        
        
    def compute_integral_loss(self, U_pred, var):
        #print(self.lw)
        #print(integrate(
        #    self.gov_eqn(U_pred, var)**2,
        #    (var, self.lb, self.ub)
        #))
        return self.lw * SymMath.efficiently_integrate(
            self.gov_eqn(U_pred, var)**2,
            (var, self.lb, self.ub)
        )
    
    '''
    Functions used to establish the initial neural network
    ===============================================================
    '''
    
    def initialize_NN(self, width, lyscl):        
        weights = []
        biases = []
        layers = [1, width, 1]
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.MPL_init(size=[layers[l], layers[l+1]], lsnow=lyscl[l])                
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=self.data_type))
            weights.append(W)
            biases.append(b)  
        return weights, biases
    
    
    def MPL_init(self, size, lsnow):
        in_dim = size[0]
        out_dim = size[1] 
        xavier_stddev = np.sqrt(2/(in_dim + out_dim)) * lsnow
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=self.data_type))
    
    
    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u
    
    def data_loss(self):
        self.u_pred = self.net_u(self.x_u) 
        loss_d = tf.reduce_mean(tf.square(self.u - self.u_pred))
        return loss_d
    
    def total_loss(self):
        return self.data_loss() + self.lw * self.eqn_loss(*self.parameters)
    
    def compute_data_gradients_tf(self):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.data_loss()
        dW = tape.gradient(loss, self.weights[0])
        dM = tape.gradient(loss, self.weights[1])
        db = tape.gradient(loss, self.biases[0])
        dc = tape.gradient(loss, self.biases[1])
        return dW, dM, db, dc
    
    def compute_eqn_gradients_sympy(self):
        dW = self.grad_W_eqn(*self.parameters)
        dM = self.grad_M_eqn(*self.parameters)
        db = self.grad_b_eqn(*self.parameters)
        dc = self.grad_c_eqn(*self.parameters)
        return dW, dM, db, dc
    
    def compute_gradients(self):
        data_grad_tf = self.compute_data_gradients_tf()
        eqn_grad_sympy = self.compute_eqn_gradients_sympy()
        total_grad = [
            data_grad_tf[i] + eqn_grad_sympy[i] for i in range(4) # collating gradients
        ]
        return total_grad
        
    
    '''
    Functions used to define ADAM optimizers
    ===============================================================
    '''
    
    # define the function to apply the ADAM optimizer
    def Adam_optimizer(self, nIter):
        varlist = self.parameters
        start_time = time.time()
        for it in range(nIter):
            gradients = self.compute_gradients()
            self.optimizer_Adam.apply_gradients(zip(gradients, varlist))
            
            # Print
            if it % 1 == 0:
                elapsed = time.time() - start_time
                loss_value = self.total_loss()
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value.numpy(), elapsed))
                start_time = time.time()
                
                
    '''
    Function used for training the model
    ===============================================================
    '''
        
    def train(self, nIter):
        self.Adam_optimizer(nIter)       
        
    def predict(self, x):
        u_p = self.net_u(x)
        return u_p
