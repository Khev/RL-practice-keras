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


class Critic:
    
    """ Critic for A2C  """
    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._make_network()
        self.lr = lr  #learning rate for optimizer
        
    def _make_network(self):
        model = Sequential()
        model.add(Dense(128,input_dim=self.input_dim, activation='relu'))
        model.add(Dense(1,activation='linear'))
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