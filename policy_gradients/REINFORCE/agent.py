""" Regular REINFORCE agent """

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
        self.model = self._make_model()
        self.lr = 0.001
        self.gamma = 0.99
        self.train = self.optimizer()
        
        #Agents memory
        self.states = []
        self.actions = []
        self.rewards = []
    
        
    def _make_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation='relu'))
        model.add(Dense(self.hidden_dim, activation = 'relu'))
        model.add(Dense(self.output_dim, activation = 'softmax'))
        return model
    
    
        
    def act(self,state):
        probs = self.model.predict(state)[0]
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
        
        #Find return
        G = self.find_discounted_return(R)
        
        #Learn
        self.train([S,A_onehot,G])

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
        
        
    def optimizer(self):
        """
        grad Loss = - mean_t (G_t * grad log pi(s_t, a_t) ) over an entire episide
        """
        
        #Placeholders
        states_pl = self.model.input
        actions_onehot_pl = K.placeholder(name='actions', shape=(None, self.output_dim))
        return_pl = K.placeholder(shape=(None,))

        #Loss
        pi_pl = self.model.output
        pi_vec = K.sum(actions_onehot_pl*pi_pl, axis = 1)
        loss_vec = -K.log(pi_vec)*K.stop_gradient(return_pl)
        loss = K.mean(loss_vec)

        #Apply updates
        opt = Adam(self.lr)
        pars = self.model.trainable_weights
        updates = opt.get_updates(loss = loss, params = pars)

        return K.function(inputs = [states_pl, actions_onehot_pl, return_pl], outputs = [], updates = updates)