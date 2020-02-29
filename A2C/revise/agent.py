""" A2C agent, where the actor is 
    
"""


import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical


class Agent:
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 16
        self.lr = 0.001
        self.gamma = 0.99
        
        #Agents memory
        self.states = []
        self.actions = []
        self.rewards = []
    
        self.actor = Actor(input_dim, output_dim, self.lr)
        self.critic = Critic(input_dim, output_dim, self.lr)
    
        
    def act(self,state):
        probs = self.actor.model.predict(state)[0]
        actions = list(range(self.output_dim))
        action = np.random.choice(actions, p = probs)
        return action
    
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def learn(self):
        
        #Sample 
        S = np.array(self.states)
        A = np.array(self.actions)
        R = np.array(self.rewards)
        
        #Change A to one-hot
        A_onehot = to_categorical(A, self.output_dim)
        
        #Find advantage
        G = self.find_discounted_return(R)
        V = self.critic.model.predict(S)
        V.resize(len(V))  #spits out a tensor
        Adv = G - V
        
        #Learn
        self.actor.train([S,A_onehot,Adv])
        self.critic.train([S,G])

        #Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        
    def find_discounted_return(self,R):
        R_discounted = np.zeros_like(R)
        running_total = 0
        for t in reversed(range(len(R_discounted))):
            running_total = running_total*self.gamma + R[t]
            R_discounted[t] = running_total
        R_discounted -= np.mean(R_discounted)
        R_discounted /= np.std(R_discounted)
        return R_discounted
    
    
    
    
#-------------------------------------------------------------------------------------------------------------------

    

class Actor:
    def __init__(self,input_dim, output_dim, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.hidden_dim = 32
        self.model = self._build_model()
        self.train = self.optimizer()
        
        
    def train(self,S, A_onehot, adv):
        self.train([S,A_onehot, adv])
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation = 'relu'))
        model.add(Dense(self.hidden_dim, activation = 'relu'))
        model.add(Dense(self.output_dim, activation = 'softmax'))
        return model
    
    
    def optimizer(self):
        """
        gradL = - E_{t} * ( Adv(t)*grad_{\theta} log(\pi(s_t, a_t)) )
        
        where E_{t} is the average over an episode
        
        """
        
        #Placeholders
        state_pl = self.model.input
        action_onehot_pl = K.placeholder(name='action_onehot', shape=(None,self.output_dim))
        adv_pl = K.placeholder(name='advantage', shape=(None,))
        
        #Set up loss
        pi_pl = self.model.output
        pi_vec = K.sum(action_onehot_pl*pi_pl, axis=1)
        loss_vec = -K.log(pi_vec)*K.stop_gradient(adv_pl)
        loss = K.mean(loss_vec)
        
        #Get updates
        opt = Adam(self.lr)
        pars = self.model.trainable_weights
        updates = opt.get_updates(loss = loss, params = pars)
        
        return K.function(inputs=[state_pl, action_onehot_pl, adv_pl], outputs = [], updates = updates)
    
    
 #-------------------------------------------------------------------------------------------------------------------

    

class Critic:
    def __init__(self,input_dim, output_dim, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.hidden_dim = 32
        self.model = self._build_model()
        self.train = self.optimizer()
        
        
    def train(self,S,G):
        self.train([S,G])
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation = 'relu'))
        model.add(Dense(self.hidden_dim, activation = 'relu'))
        model.add(Dense(1, activation = 'linear'))
        return model
    
    
    def optimizer(self):
        """
           L = E_t()
        """
        
        #Placeholders
        S_pl = self.model.input
        V_pl = self.model.output
        G_pl = K.placeholder(name='discounted_return', shape=(None,))
        
        #loss
        loss = K.mean( K.square(V_pl - G_pl) )
        
        #Get updates
        opt = Adam(self.lr)
        pars = self.model.trainable_weights
        updates = opt.get_updates(loss = loss, params = pars)
        
        return K.function(inputs=[S_pl, G_pl], outputs = [], updates = updates)