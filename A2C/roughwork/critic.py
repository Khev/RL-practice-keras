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
        
        #optimizer
        self.opt = self.optimizer()
        
        
    def learn(self,S,adv):
        self.opt([S,adv])

        
    def _make_network(self):        
        S = Input(shape=(self.input_dim,))
        x = Dense(128, activation = 'relu')(S)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = S, outputs = out)
        model.compile(loss = 'mse', optimizer = Adam( lr = self.lr, clipnorm = self.clipnorm))
        return model
    
       
    def optimizer(self):
        """ The critic loss: mean squared error over discounted rewards """
        
        #Placeholders
        discounted_returns_placeholder = K.placeholder(name='discounted_return',shape=(None,))        
        critic_loss = K.mean(K.square(discounted_returns_placeholder - self.model.output))
        
        #Define optimizer
        adam_critic = RMSprop(lr = self.lr, epsilon = 0.1, rho = 0.99)  #arbitray
        pars = self.model.trainable_weights
        updates = adam_critic.get_updates(params=pars,loss=critic_loss)
        
        return K.function([self.model.input, discounted_returns_placeholder], [], updates=updates)        
        
 
    def learn1(self,S,G):
        
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