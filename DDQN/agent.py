""" DQN Agent -- implemented following https://www.nature.com/articles/nature14236
    
    with (a) experience replay
         (b) target network, which I update every C timesteps
         
    I have NOT done gradient clipping, or reward scaling.
"""


import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical



class Agent:
       
    def __init__(self,input_dim, output_dim, lr, gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.2
        self.memory = []
        self.batchsize = 32   #this is what google picked in their Nature paper -- copy them!
        self.memory_size = 2000
        
        #Make neural nets
        self.model = self._make_model()
        self.target_model = self._make_model()      #we keep a target model which we update every K timesteps
        self.C = 1000                               # this stabilizes learning
        self.tau = 0.1                              # this is the soft update parameter -- see 'def update_target_network'                             
    
    
    def _make_model(self):
        
        model = Sequential()
        model.add( Dense(64, input_dim = self.input_dim, activation='relu') )
        model.add( Dense(64, activation='relu') )
        model.add( Dense( self.output_dim, activation = 'linear' ))
        model.compile(loss='mse',optimizer=Adam(lr = self.lr))
 
        return model
                  
                  
    def remember(self, state, action, reward, next_state, done):
        event = (state,action,reward, next_state, done)
        if len(self.memory) <= self.memory_size:
            self.memory.append(event)
        else:
            self.memory[0] = event
                  

    def act(self, state):
        """ epsilon greedy """
        
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            Qs = self.model.predict(state)[0]   #keras returns a tensor (2d array), so take first element
            action = np.argmax(Qs)
                  
        return action
                  
                  
    def replay(self):
        """ Does experience replay. First trains the behavior network (as opposed to the target network)
            using a minibatch from the buffer. The Q_want is found using the target network
            
            Q_want(state,action) = reward + gamma*max_{action'} Q_target(next_state,action'), for non-terminal next_state
                                   = reward                                             , for terminal next_state
        """
        
        #grab random batch
        if len(self.memory) < self.batchsize:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory,self.batchsize)
              
        #instantiate
        states = []
        Q_wants = []
        
        #Find updates
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
            # Doing this -- i.e. finding a* from the behavior network, is what 
            # distinguihses DDQN from DQN
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
        
    
    def hard_update_target_network(self,step):
        """ Here the target network is updated every K timesteps
            By update, I mean clone the behavior network.
        """
        
        if step % self.C == 0:
            pars = self.model.get_weights()
            self.target_model.set_weights(pars)
            
        
    def soft_update_target_network(self):
        """  Here the target network is updated every timestep,
             according to 
             
             theta_target = (1-tau)*theta_target + theta_behavior*tau
             
             where,
             tau = parameter (the smaller, the softer)
             theta_target = parameters from the target network
             theta_behaviour = parameters from the behavior network
  
        """
        
        pars_behavior = self.model.get_weights()       # these have form [W1, b1, W2, b2, ..], Wi = weights of layer i
        pars_target = self.target_model.get_weights()  # bi = biases in layer i
        
        ctr = 0
        for par_behavior,par_target in zip(pars_behavior,pars_target):
            par_target = par_target*(1-self.tau) + par_behavior*self.tau
            pars_target[ctr] = par_target
            ctr += 1

        self.target_model.set_weights(pars_target)