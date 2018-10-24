import numpy as np

class Env:
    def __init__(self,num_states):
        self.observation_space = num_states
        self.action_space = 2
        
        
    def step(self,state,action):
        
        #Convert state from 1-hot to scalar
        state_scalar = np.argmax(state)   
        
        #Find next state
        if action == 1:
            next_state = state_scalar + 1
        else:
            next_state = 0
        
        #Check for terminal state
        done,reward = False,0
        if next_state == self.observation_space - 1:
            reward, done = 1, True
            
        return next_state, reward, done, 0  #0 is for info; including it by convention

    
    def reset(self):
        return 0