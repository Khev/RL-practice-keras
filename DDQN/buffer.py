""" Buffer """


import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Buffer:
       
    def __init__(self,input_dim, output_dim, lr, gamma):
        self.S = []
        self.A = []
        self.R = []
        self.S1 = []
        self.D = []
        self.TD = []  #TD error
        self.batchsize = 32   #this is what google picked in their Nature paper -- copy them!
        self.memory_size = 2000
                                 
    
    
       
                  
    def remember(self, state, action, reward, next_state, done):
        """ Add experience to buffer """

        
        #Find TD error
        First find action from behavior network
        Q_next = self.model.predict(next_state)[0]
        action_next_best = argmax(Q_next)

        #Then find the yi
        Q_next_target = self.target_model.predict(next_state)[0]
        yi = reward + (1-done)*self.gamma*Q_next_target(action_next_best)
        
        #Define td_error
        td_error = abs(yi - Q_next)

        #Add event to memory
        if len(self.S) <= self.memory_size:
            self.S.append(state)   
            action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
            self.A.append(action_onehot)
            self.R.append([reward])
            self.S1.append(next_state)
            self.D.append(done)
            self.TD.append([td_error])
            
        else:
            self.S[0] = state   
            action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
            self.A[0] = action_onehot
            self.R[0] = [reward]
            self.S1[0] = next_state
            self.D[0] = done
            self.TD[0] = [td_error]
            
            
    def get_batch(self, prioritised = False):
        
        if prioritised == False:
            indicies = np.random.choice(range(len(self.S)),self.batchsize)
            S = np.array(self.S)[indicies]
            A = np.array(self.A)[indicies]
            R = np.array(self.R)[indicies]
            S1 = np.array(self.S1)[indicies]
            D = np.array(self.D)[indicies]
            TD = np.array(self.TD)[indicies]
            
        else:
            #Sample greedily according to TD error
            probs = self.TD
            probs = np.array(probs) / (1.*sum(probs))
    
            #Then do the sampling
            indicies = np.random.choice(range(len(self.S)),self.batchsize, p=probs)
            S = np.array(self.S)[indicies]
            A = np.array(self.A)[indicies]
            R = np.array(self.R)[indicies]
            S1 = np.array(self.S1)[indicies]
            D = np.array(self.D)[indicies]
            TD = np.array(self.TD)[indicies]
            
        return S,A,R,S1,D,TD
                  
                