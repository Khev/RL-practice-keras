import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Critic:
    
    """ Critic for A2C  """
    
    def __init__(self,input_dim, output_dim,lr,gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr                                #learning rate for optimizer
        self.gamma = gamma                          #discount factor
        self.model = self._make_network()
        self.target_model = self._make_network()    #we have target networks to stabilize learning.
        pars = self.model.get_weights()
        self.target_model.set_weights(pars)         #clone the network
        self.find_action_grads = self._make_gradient_function()
       
             
    def _make_network(self):
        """ Q(s,a) -- so two arguments. I'll concatenate
            the s and a vectors. I'll use the 'Input' 
            formalism for models here, as opposed to the
            Sequential one I did.
        """
        
        state = Input(shape=(self.input_dim,))
        action = Input(shape=(self.output_dim,))
        
        #Define with state first
        x = Dense(256, activation = 'relu')(state)
        x = concatenate([x,action])
        x = Dense(128, activation = 'relu')(x)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = [state,action], outputs = out)
        model.compile(loss='mse',optimizer=Adam(self.lr))
        return model 
       
    
    def _make_gradient_function(self):   
        
        #Inputs
        state_pl = self.model.input[0]     #pl = placeholder, these are inputs in the keras functional API
        action_pl = self.model.input[1]
        
        #Outputs
        Q = self.model.output
        action_grad_Q = K.gradients(Q,action_pl)[0]
        
        return K.function(inputs = [state_pl, action_pl], outputs = [action_grad_Q])
    
    
      
    def learn(self,minibatch):
        
        """ The critic target is just the usual 
        
            Q(s_t,a_t) = r_t + \gamma*  Q_target(s'_t, a'_t)   for s' non-terminal
            Q(s_t, a_t) = r_t                                  for s' terminal
            
            (I'll call the RHS Q_want)
            
      
            Input: batch = [(s,a,r,s',done), ...]
            
        """
        
        states = []
        actions = []
        Q_wants = []
        for event in minibatch:
            state,action,reward,next_state,done = event
            states.append(state)
            actions.append(action)
            
            if done == True:
                Q_want = reward
            else:
                state_tensor, action_tensor = np.reshape(state,(1,len(state))), np.reshape(action,(1,len(action)))
                Q_target_next = self.target_model.predict([state_tensor,action_tensor])[0]   
                Q_want = reward + self.gamma*Q_target_next
            Q_wants.append(Q_want)
           
        
        #Here I fit on the whole batch. Others seem to fit line-by-line
        #Dont' think (hope) it makes much difference
        states = np.array(states)
        actions = np.array(actions)
        Q_wants = np.array(Q_wants)        
        self.model.train_on_batch([states, actions],Q_wants)