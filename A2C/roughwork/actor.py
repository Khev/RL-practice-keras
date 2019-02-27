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
    
    """ Actor for A2C """
    
    def __init__(self,input_dim, output_dim,lr, gamma, tau, clipnorm, verbose = False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._make_network()
        self.target_model = self._make_network()    #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())         #clone the network
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.tau = tau
        self.c_entropy = 0.001
        self.clipnorm = clipnorm
        self.verbose = verbose
        self.opt = self.optimizer()
        
        
    def _make_network(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        return model
    
    
    def learn(self,S,A,adv):
        self.opt([S,A,adv])

        
    def optimizer(self):
        
        """ Loss =  (r + gamma*V(s1) - V(s)) * grad_(theta) ln( \pi(s,a) )
        
            where
            s = state
            s1 = next state
            a = action
            \pi(s,a) = the policy (actor)
            V(.) = the value from the critic
        """
        
        # taken from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
        # I believe this is a work around to get keras to learn **given a gradient**
        # As opposed to bunch of x_train, y_trains?
        
        #Inputs
        state_pl = self.model.input
        A_onehot_pl = K.placeholder(name='A_onehot', shape=(None,self.output_dim))
        adv_pl = K.placeholder(name='adv', shape=(None,1))

        #Regular loss
        pi_pl = self.model.output  # (N_samples, out_dim)
        pi_specific_action = K.sum(A_onehot_pl*pi_pl ,axis=1)  #(N_samples)
        loss_regular = K.mean(K.log(pi_specific_action)*K.stop_gradient(adv_pl))  #scalar               
        
        #Entropy loss
        loss_entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        
        #Total loss
        loss = loss_regular + self.c_entropy*loss_entropy 
        
        #Compute gradietn
        adam = Adam(self.lr)
        pars = self.model.trainable_weights
        updates = adam.get_updates(loss = loss, params = pars)

        return K.function(inputs = [state_pl, A_onehot_pl, adv_pl], outputs = [], updates = updates)