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
        self.clipnorm = clipnorm
        self.verbose = verbose
        self.opt = self.optimizer()
        
        
    def _make_network(self):
        #model = Sequential()
        #model.add(Dense(128,input_dim=self.input_dim, activation='relu'))
        #model.add(Dense(self.output_dim,activation='softmax'))
        #return model
    
        # Neural Net -- these were the states I originally used (as opposed to the one above, which was from the external code)
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        return model
    
    
    def learn(self,S,Q):
        self.opt([S,Q])
    
    
    def optimizer(self):
        
        """ Loss =  Q(s,a) * grad_(theta) ln( \pi(s,a) )
        
            where
            s = state
            a = action
            \pi(s,a) = the policy (actor)
            Q(s,a) = the Q-value, from the critic
        """
        
        # taken from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
        # I believe this is a work around to get keras to learn **given a gradient**
        # As opposed to bunch of x_train, y_trains?
        
        #Inputs
        state_pl = self.model.input
        Q_pl = K.placeholder(shape=(None,1))  
                                        
        #Find grad log( pi )
        pi_pl = self.model.output
        pars = self.model.trainable_weights
        grads = tf.gradients(pi_pl, pars, -Q_pl)   #scalar multiply by the Q
        
        #grads_and_pars = zip(pars_grad_mu, pars)  #keras needs this form
        #updates = tf.train.AdamOptimizer(self.lr).apply_gradients(grads_and_pars)

        # The gradients as defined above work on my mac, but not ubuntu.
        # Below I am trying a workaround. I changed the keras source code 
        # To get this working. Specifically, I make the optimizer.get_updates()
        # function accept custom gradients. It was easy to do.
        
        opt = Adam(self.lr)
        loss = grads  #placeholder, I won't use it
        updates = opt.get_updates(loss = grads, params = pars, grads = grads)

        return K.function(inputs = [state_pl, Q_pl], outputs = [], updates = updates)
        #return K.function(inputs = [state_pl, action_graQ_plds_pl], outputs = [updates])