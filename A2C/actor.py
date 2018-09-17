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
    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._make_network()
        self.lr = lr  #learning rate for optimizer
        
        
    def _make_network(self):
        #model = Sequential()
        #model.add(Dense(128,input_dim=self.input_dim, activation='relu'))
        #model.add(Dense(self.output_dim,activation='softmax'))
        #return model
    
        # Neural Net -- these were the states I originally used (as opposed to the one above, which was from the external code)
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        return model
    
    
    def optimizer(self):
        """ The actor loss: mean_t ( log(pi(s_t,a_t)) * A_t ), 
        
            where A_t = advantage,  A_t = G_t - V(s_t), G_t 
          
            where V(s) = value of state
            and G_t = discounted return at time t
        
            Following (Cf. https://arxiv.org/abs/1602.01783), we add an entropy
            term to the loss, to encourage exploration. 
        """
        
        #Place holders (think of these as function inputs)
        state_placeholder = self.model.input
        all_probs_placeholder = self.model.output
        actions_onehot_placeholder = K.placeholder(name='actions_onehot',shape=(None,self.output_dim))
        advantage_placeholder = K.placeholder(name='discounted_return',shape=(None,))
        
        #Find log-probs
        action_probs = K.sum(actions_onehot_placeholder*all_probs_placeholder ,axis=1)  #get prob for specific action
        log_probs = K.log(action_probs) 
        
        #I want to the gradient to by on the log(pi) only, so 'protect' the advantage
        # This is the way keras works -- I think!
        eligibility = log_probs*K.stop_gradient(advantage_placeholder)
        
        #Add in entropy -- see (Cf. https://arxiv.org/abs/1602.01783)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)


        loss = 0.001*entropy - K.mean(eligibility)
        
        #Define optimizer
        adam = Adam()
        pars = self.model.trainable_weights
        updates = adam.get_updates(params=pars,loss=loss)
    
        #Then return
        return K.function([state_placeholder, actions_onehot_placeholder, advantage_placeholder],[],updates=
                         updates)