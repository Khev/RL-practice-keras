import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Input, concatenate


class Critic:
    
    """ Critic for A2C  """
    
    def __init__(self,input_dim, output_dim,lr,gamma,tau, clipnorm, verbose = False):
        
        #Pars
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.tau = tau
        self.verbose = verbose
        self.clipnorm = clipnorm
        
        #Make models
        self.model = self._make_network()
        self.target_model = self._make_network()                        #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())         #clone the network

        
    def _make_network(self):
        """ Q(s,a) -- function of two arguments """
        
        S = Input(shape=(self.input_dim,))
        x = Dense(128, activation = 'relu')(S)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = S, outputs = out)
        model.compile(loss = 'mse', optimizer = Adam( lr = self.lr, clipnorm = self.clipnorm))
        return model
       
 
    def learn(self,S,R,D,V1,G):
        
        """ 
            The loss function for the critic is
           
            L_i = \sum_{batch}  ( V_i - y_i )^2 
            
            Where
            
            y_i = r_i + (1-done) gamma* max( V_i(s', a') )  for non-terminal \vec{x'}
        
           And,
           
           r_i = reward to agent i
           gamma = discount factor
           s' = next state
           a' = most probable action in the next state
           
        """  

                
        #Find yi
        #yi = R + (1-D)*self.gamma*V1
                    
        #Train
        #loss = self.model.train_on_batch(S, yi)
        loss = self.model.train_on_batch(S, G)
        if self.verbose == True:
            print 'critic loss = ' + str(loss)
        return 