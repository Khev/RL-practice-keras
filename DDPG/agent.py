import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical

#Sub classes
from actor import Actor
from critic import Critic


class Agent:
       
    def __init__(self,input_dim, output_dim, lr, gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.tau = 0.1
        
        #For experience replay
        self.memory = []
        self.memory_size = 10000
        self.batchsize = 32 
        
        #Actor & critic
        self.actor = Actor(input_dim,output_dim,self.lr)
        self.critic = Critic(input_dim,output_dim, self.lr, self.gamma)
        
        
    def remember(self, state, action, reward, next_state, done):
        event = (state,action,reward, next_state, done)
        if len(self.memory) <= self.memory_size:
            self.memory.append(event)
        else:
            self.memory[0] = event
         

    def act(self, state):        
        action =  self.actor.model.predict(state)[0]
        return action
                
        
    def extract_from_batch(self,batch):
        states, actions = [], []
        for event in batch:
            state,action,reward,next_state,done = event
            states.append(state)
            actions.append(action)
        return np.array(states), np.array(actions)
            
    
    def train_models(self):
       
        #Do experience replay
        if len(self.memory) < self.batchsize:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory,self.batchsize)
            
            
        #Actor update
        states, actions = self.extract_from_batch(minibatch)
        grad_actions = self.critic.find_action_grads([states,actions])[0]
        self.actor.learn(states,grad_actions)
        self.soft_update_target_network(self.actor)
        
        #Critic update
        self.critic.learn(minibatch)
        self.soft_update_target_network(self.critic)
        
        
    def soft_update_target_network(self,net):
        """
        Updates parameters of the target network according to the following
        where tau is a hyper parameter.
        
        theta_target = (1-tau)*theta_target + tau*theta_behavior
        
        Input: network = Actor or Critic class
        """
        
        pars_behavior = net.model.get_weights()       # these have form [W1, b1, W2, b2, ..], Wi = 
        pars_target = net.target_model.get_weights()  # bi = biases in layer i
        
        ctr = 0
        for par_behavior,par_target in zip(pars_behavior,pars_target):
            par_target = par_target*(1-self.tau) + par_behavior*self.tau
            pars_target[ctr] = par_target
            ctr += 1

        net.target_model.set_weights(pars_target)
                 

