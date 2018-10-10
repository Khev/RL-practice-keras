import numpy as np
import tensorflow as tf
import random
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
from tensorflow import set_random_seed
from numpy.random import seed

#Sub classes
from actor import Actor
from critic import Critic


class Agent:
       
    def __init__(self,input_dim, output_dim, lr, gamma, seed_num = False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.tau = 0.1
        self.seed_num = seed_num
        
        #For experience replay
        self.memory = []
        self.memory_size = 10000
        self.batchsize = 32 
        
        #Actor & critic
        self.actor = Actor(input_dim,output_dim,self.lr)
        self.critic = Critic(input_dim,output_dim, self.lr, self.gamma)
        
        if seed_num != False:
            set_random_seed(seed_num)  #seed tensorflow
            seed(seed_num)             #seed numpy


        
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
        
        
    def save_target_weights(self):

        """ Saves the weights of the target 
            network (only use the target during
            testing, so don't need to save tje
            behavior)
        """

        #Create directory if it doesn't exist
        dir_name = 'network_weights/'
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))

        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau)
        else:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau) \
            +'_seed_' + str(self.seed_num)
            

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = self.actor.target_model.get_weights()
        np.save(filename + pars_tag, actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = self.critic.target_model.get_weights()
        np.save(filename + pars_tag, critic_pars)




    def load_target_weights(self,gamma,lr,tau):

        """ Loads the weights of the target 
            network, previously created using
            the save_target_wieghts() function
        """


        #Now save the weights. I'm choosing ID by gamma, lr, tau
        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_' + str(self.tau) + '.npy'
        else:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_'+str(self.tau)+'_seed_' +str(self.seed_num)+ '.npy'

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = np.load(filename + pars_tag)
        self.actor.target_model.set_weights(actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = np.load(filename + pars_tag)
        self.critic.target_model.set_weights(critic_pars)
                 

