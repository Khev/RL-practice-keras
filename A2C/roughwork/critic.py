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
        """ The critic loss: L_i = \sum_{batch}  ( V_i - G_i )^2 
        
            where G_i is the discounted reward.
        
        """
        
        #Placeholders
        discounted_returns_placeholder = K.placeholder(name='discounted_return',shape=(None,))        
        critic_loss = K.mean(K.square(discounted_returns_placeholder - self.model.output))
        
        #Define optimizer
        adam_critic = RMSprop(lr = self.lr, epsilon = 0.1, rho = 0.99)  #arbitray
        pars = self.model.trainable_weights
        updates = adam_critic.get_updates(params=pars,loss=critic_loss)
        
        return K.function([self.model.input, discounted_returns_placeholder], [], updates=updates)        
        
 
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
        self.target_model = self._make_network()                       
        self.target_model.set_weights(self.model.get_weights()) 
        
        #optimizer
        self.opt = self.optimizer()
        
        
    def learn(self,S,R,D,V1):
        V1 = self.opt([S,R,D,V1])
        return V1
    
    
    def _make_network(self):        
        S = Input(shape=(self.input_dim,))
        x = Dense(128, activation = 'relu')(S)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = S, outputs = out)
        model.compile(loss = 'mse', optimizer = Adam( lr = self.lr, clipnorm = self.clipnorm))
        return model
       
 
    def optimizer(self):
        
        """ 
            The loss function for the critic is
           
            L_i = \sum_{batch}  ( V_i - y_i )^2 
            
            where,
            
            y_i = r_i + (1-done) gamma* V_i(s)  for non-terminal \vec{x'}
            r_i = reward to agent i
            gamma = discount factor
            done = 1 if episode finished, 0 otherwise
        """
        
        #Placeholders (think of these as inputs)
        S_pl = self.model.input
        V_pl = self.model.output
        R_pl = K.placeholder(name='reward',shape=(None,))  #assumes R has form np.array([[reward1], [reward2], ..])
        D_pl = K.placeholder(name='done', shape=(None,))   #assumes D has form np.array([[D1], [D2], ..])
        V1_pl = K.placeholder(name='V1',shape=(None,))

        #Find yi
        V1 = K.sqrt(K.square(V1_pl))
        Y = R_pl + (1.0-D_pl)*self.gamma*V1_pl  #1D array
        #Y = np.array([ [i] for i in Y])
        
        #Find loss
        loss = K.mean(K.square(V_pl - Y))     #scalar
        
        #Define optimizer
        adam_critic = RMSprop(lr = self.lr, epsilon = 0.1, rho = 0.99)  #arbitray
        pars = self.model.trainable_weights
        updates = adam_critic.get_updates(params=pars,loss=loss)
        
        return K.function([S_pl, R_pl, D_pl,V1_pl], [], updates=updates)  