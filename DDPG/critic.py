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
        self.lr = lr  #learning rate for optimizer
        self.model = self._make_network()
        self.target_model = self._make_network()    #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())  #clone the networks
        
        # K.function( input = [s,a] ), output = grad_a Q(s,a)
        action_placeholder = K.placeholder(shape=(None,self.output_dim))
        self.action_grads = K.function([self.model.input, action_placeholder],
                                       K.gradients(self.model.output,self.model.input))

             
    def _make_network(self):
        model = Sequential()
        model.add(Dense(64,input_dim=self.input_dim, activation='relu'))
        model.add(Dense(64,input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.output_dim,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr = self.lr))
        return model
    
    
    def action_gradients(self,states,actions):
        return self.action_grads([states,actions])
    
      
    def learn(self,minibatch):
        
        """ The critic target is just the usual 
        
            Q(s_t,a_t) = r_t + \gamma* max_(a) Q(s'_t, a'_t)
            
            Input: batch = [(s,a,r,s',done), ...]
            
        """
        
        
        for event in minibatch:
            state,action,reward,next_state,done = event
            states.append(state)
            
            #Find Q_target
            state_tensor = np.reshape(state,(1,len(state)))  # keras takes 2d arrays
            Q_want = self.model.predict(state_tensor)[0]     # all elements of this, except the action chosen, stay
                                                             # the same                       
            
            #If state is terminal, Q_target(action) = reward
            if done == True:
                Q_want[action] = reward
            
            # Q_want(action) = reward + gamma*max_a Q_target(next_state, a*)  -- note I sample from the target network
            # where a* = argmax Q(next_state,a)  
            else:
                next_state_tensor = np.reshape(next_state,(1,len(next_state)))  
                Q_next_state_vec = self.model.predict(next_state_tensor)
                action_max = np.argmax(Q_next_state_vec)
                
                Q_target_next_state_vec = self.target_model.predict(next_state_tensor)[0]
                Q_target_next_state_max = Q_target_next_state_vec[action_max]
                
                Q_want[action] = reward + self.gamma*Q_target_next_state_max
                Q_want_tensor = np.reshape(Q_want,(1,len(Q_want)))
                #self.model.fit(state_tensor,Q_want_tensor,verbose=False,epochs=1)
            
            Q_wants.append(Q_want)
        
        #Here I fit on the whole batch. Others seem to fit line-by-line
        #Dont' think (hope) it makes much difference
        states = np.array(states)
        Q_wants = np.array(Q_wants)
        self.model.fit(states,Q_wants,verbose=False, epochs=1)