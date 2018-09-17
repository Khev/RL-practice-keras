import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Actor:
    
    """ Actor for DDPG """
    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr                                            # learning rate for optimizer
        self.model = self._make_network()
        self.target_model = self._make_network()                # target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights()) # clone the networks
        self.adam_optimizer = self.optimizer()
        
        
    def _make_network(self):
        # Neural Net
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='softmax'))  #recall, deterministic policy, so only 1 output
        return model
    
    
    def learn(self,states,grad_actions):
        self.adam_optimizer([states,grad_actions]) 
    
    
    def optimizer(self):
        """ Loss = grad_(theta) \mu(s)  grad_(actions) Q(s,a)
        
            where
            s = state
            a = action
            \mu_(\theta)(s) = the deterministic policy (actor)
            Q(s,a) = the Q-value, from the critic
        """
        
        # taken from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
        # I believe this is a work around to get keras to learn _given a gradient_
        # As opposed to bunch of x_train, y_trains?
        
        grad_actions = K.placeholder(shape=(None,self.output_dim))
        grad_pars = tf.gradients(self.model.output,self.model.trainable_weights, -grad_actions)
        grads = zip(grad_pars, self.model.trainable_weights)
        return K.function([self.model.input, grad_actions], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])
      