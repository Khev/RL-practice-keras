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
        values = self.critic.model.predict(self.states)
        values.resize(len(values))
        advantages = discounted_return - values
        
        #Do the training
        self.train_actor([self.states,self.action_samples,advantages])
        self.train_critic([self.states,discounted_return])
        
        #Erase memory
        self.states = []
        self.action_samples = []
        self.rewards = []
        
        
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
        """ Choose action according to softmax """
        
        probs =  self.actor.model.predict(state)[0]
        action = np.random.choice(self.actions, p=probs)
        return action