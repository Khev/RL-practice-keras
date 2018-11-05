""" DDQN Agent with prioritised replay """


import numpy as np
import tensorflow as tf
from buffer import Buffer
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
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
        self.batchsize = 32  
        self.memory_size = 2000
        self.buffer = Buffer(output_dim, self.memory_size, self.batchsize)
        
        #Make neural nets
        #self.model = self._make_model()
        self.model = self._make_model_atari()
        #self.target_model = self._make_model()     
        self.target_model = self._make_model_atari()
        self.target_model.set_weights(self.model.get_weights()) # clone the networks
        self.C = 1000                               # hard update parameter
        self.tau = 0.1                              # this is the soft update parameter -- see 'def update_target_network'                             
    
    
    def _make_model(self):
        
        model = Sequential()
        model.add( Dense(64, input_dim = self.input_dim, activation='relu') )
        model.add( Dense(64, activation='relu') )
        model.add( Dense( self.output_dim, activation = 'linear' ))
        model.compile(loss='mse',optimizer=Adam(lr = self.lr))
 
        return model


    #Taken from https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    def _make_model_atari(self):
        
        n_actions = self.output_dim
        
        # We assume a theano backend here, so the "channels" are first.
        ATARI_SHAPE = (4, 105, 80)

        # With the functional API we need to define the inputs.
        frames_input = Input(ATARI_SHAPE, name='frames')
        actions_input = Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0)(frames_input)

        conv_1 = Convolution2D(32, (3, 3), activation='relu', data_format='channels_first')(normalized)
        conv_2 = Convolution2D(32, (3, 3), activation='relu')(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        #filtered_output = merge([output, actions_input], mode='mul')
        filtered_output = concatenate([output, actions_input])
        model = Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')
 
        return model
                  
                  
    def remember(self, state, action, reward, next_state, done):
        
        #Find TD error:  abs(Q(s,a) - yi)
        #First find action from behavior network
        Q_next = self.model.predict(self.make_tensor(next_state))[0]
        action_next_best = np.argmax(Q_next)

        #Then find the yi
        Q_next_target = self.target_model.predict(self.make_tensor(next_state))[0]
        yi = reward + (1-done)*self.gamma*Q_next_target[action_next_best]
        
        
        #Then find Q(s,a)
        Q_vec = self.target_model.predict(self.make_tensor(state))[0]
        action_scalar = np.argmax(Q_vec)  #actions are stored as 1 hots
        Q = Q_vec[action_scalar]
        
        #Define td_error
        td_error = abs(yi - Q)
        
        self.buffer.remember(state,action,reward,next_state,done,td_error)
                  

    def act(self, state):
        """ epsilon greedy """
        
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            Qs = self.model.predict(state)[0]   #keras returns a tensor (2d array), so take first element
            action = np.argmax(Qs)
                  
        return action
                  
                  
    def replay(self, prioritised = False):
        """ Does experience replay.
        """
        
        #grab random batch
        S,A,R,S1,D,TD = self.buffer.get_batch(prioritised = False)
              
        #instantiate
        states = []
        Q_wants = []
        
        #Find updates
        for i in range(len(S)):
            state,action,reward,next_state,done,td = S[i],A[i],R[i],S1[i],D[i],TD[i]
            action = np.argmax(action)  #convert from 1-hot
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
        
        
        
    def make_tensor(self,vec):
        """ Reshapes a 1-hot vector
            into a 1-hot tensor -- keras requires the tensor
        """
        
        return np.reshape(vec, (1,len(vec)))