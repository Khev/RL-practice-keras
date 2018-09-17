import numpy as np
import tensorflow as tf
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
        
        #These will store the samples from which the agent will learn
        self.states = []
        self.action_samples = []
        self.rewards = []
        
        #Make actor and critic
        self.actor = Actor(input_dim,output_dim,self.lr)
        self.critic = Critic(input_dim,output_dim, self.lr)
        
        self.train_actor = self.actor.optimizer()
        self.train_critic = self.critic.optimizer()
            
    
    def train_models(self):
        
        #Turn lists into arrays
        self.states = np.array(self.states)
        self.action_samples = np.array(self.action_samples)
        self.rewards = np.array(self.rewards)
        
        #Compute inputs for the optimizers
        discounted_return = self.find_discounted_return(self.rewards)
        Q_values = self.critic.model.predict(self.states)
        Q_values.resize(len(values))
        advantages = discounted_return - Q_values
        
        #Do the training
        self.train_actor([self.states,self.action_samples,advantages])
        self.train_critic([self.states,discounted_return])
        
        #Update the target networks
        self.soft_update_target_network(self.actor)
        self.soft_update_target_network(self.critic)
                
        #Erase memory
        self.states = []
        self.action_samples = []
        self.rewards = []
        
        
        
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
                
        
    def find_discounted_return(self,rewards):
        R = np.zeros_like(rewards)
        rolling_sum = 0
        for t in reversed(range(len(R))):
            rolling_sum = rolling_sum*self.gamma + rewards[t]
            R[t] = rolling_sum
            
        #Normalize rewards
        R -= np.mean(R)
        R /= np.std(R)
            
        return np.array(R)
    
        
    
    def remember(self, state, action, reward):
        self.states.append(state)
        
        action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
        self.action_samples.append(action_onehot) 
        
        self.rewards.append(reward)
        

    def act(self, state):        
        action =  self.actor.model.predict(state)[0]
        return action[0][0]
