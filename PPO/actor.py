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
    """ Actor for PPO - https://arxiv.org/pdf/1707.06347.pdf
    
        I used code from https://github.com/FitMachineLearning/PPO-Keras/blob/master/ppo.py
        as a reference
    """
    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._make_network()
        self.lr = lr  #learning rate for optimizer
        self.eps = 0.2  #for loss clipping
        
        
    def _make_network(self):
        #model = Sequential()
        #model.add(Dense(128,input_dim=self.input_dim, activation='relu'))
        #model.add(Dense(self.output_dim,activation='softmax'))
        #return model
    
        # Neural net
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        return model
    
    
    def optimizer(self):
        """ The actor loss: min( r(\theta)*A_t, clip(r(\theta), (1-epsilon, 1+epsilon))*A_t ),
  
  
            where r(\theta) = \pi_{\theta} / \pi_{\theta_old}
            
                  A_t = advantage,  A_t = G_t - V(s_t) 
          
            where V(s) = value of state
            and G_t = discounted return at time t
            and mean_t is the empirical average over a batch of experiences: mean( (S,A,R)_1, (S,A_R)_2, ... )
            
            Note: my actor and critic do NOT share a NN, so my loss function is different to  eq (9) in
            the paper (exclude the L^{VF} term. )
            
        
            Following (Cf. https://arxiv.org/abs/1602.01783), we add an entropy
            term to the loss, to encourage exploration. 
        """
        
        
        #Placeholders, think of these are inputs
        state_placeholder = self.model.input
        pi_all_placeholder = self.model.output    # \pi for all actions 
        pi_all_old_placeholder = self.model.output
        actions_onehot_placeholder = K.placeholder(name='actions_onehot',shape=(None,self.output_dim))
        advantage_placeholder = K.placeholder(name='discounted_return',shape=(None,))
        
        #Find terms for loss
        pi_specific = K.sum(action_onehot_placeholder*pi_all_placeholder,axis=1)           #select \pi for the action taken
        pi_specific_old = K.sum(action_onehot_placeholder*pi_all_old_placeholder,axis=1)
        r = pi_specific / pi_specific_old
        
        term1 = r*advantage_placeholder
        term2 = K.clip(r,min_value = 1 - self.eps, max_value= 1 + self.eps)*advantage_placeholder
        temp = K.minimum(term1,term2)
        
        #Add in entropy to encourage exploration -- see (Cf. https://arxiv.org/abs/1602.01783)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        
        loss = -( K.mean(temp) + 0.001*entropy)  #0.001 is obv arbitrary.
        
        
        #Define the optimizer
        adam = Adam()
        pars = self.model.trainable_weights
        updates = adam.get_updates(params=pars,loss=loss)
        
        #Put altogether -- this is the syntax for the keras functional API 
        return K.function(inputs=[state_placeholder,actions_onehot_placeholder,advantage_placeholder], outputs =[],
                         updates = updates)