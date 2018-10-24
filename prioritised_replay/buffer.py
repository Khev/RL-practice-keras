""" Buffer """

import numpy as np
from keras.utils import to_categorical


class Buffer:
       
    def __init__(self,output_dim,memory_size,batchsize):
        self.S = []
        self.A = []
        self.R = []
        self.S1 = []
        self.D = []
        self.TD = []  #TD error
        self.batchsize = batchsize  
        self.memory_size = memory_size
        self.output_dim = output_dim   #need this to convert action into 1-hots
                                 
                  
    def remember(self, state, action, reward, next_state, done, td_error):
        """ Add experience to buffer """

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
            probs = self.TD.flatten()
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
    
    
    def num_memories(self):
        return len(self.S)
                  
                