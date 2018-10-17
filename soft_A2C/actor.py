import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils import to_categorical


class Actor:
        
    def __init__(self,input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose = False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._make_network()
        self.target_model = self._make_network()    #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())         #clone the network
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.clipnorm = clipnorm
        self.verbose = verbose
        self.opt = self.optimizer()
        
        
    def _make_network(self):
        #model = Sequential()
        #model.add(Dense(128,input_dim=self.input_dim, activation='relu'))
        #model.add(Dense(self.output_dim,activation='softmax'))
        #return model
    
        # Neural Net -- these were the states I originally used (as opposed to the one above, which was from the external code)
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        return model
    
    
    def learn(self,S,A,Q,V):
        """ 
        S = batch of states
        A = batch of actions
        Q = batch of Q-vals
        V = batch of values
        """
        
        self.opt([S,A,Q,V])
    
    
    def optimizer(self):
        
        """ grad_loss =  \grad_{theta} \log(\pi(a_t,s_t))* (  \alpha \log(\pi(s_t, a_t) ) - Q(s_t, a_t)  + V(s_t)  )
        
            where
            s_t = state at time t (t is discrete)
            a_t = action
            \pi(s,a) = the policy (actor)
            Q(s,a) = the Q-value, from the critic
            V(s) = the value  (gets its own neural net -- see paper)
            \alpha = parameter controlling the strength of the entropy
            
        """
        
        
        #Inputs
        state_pl = self.model.input
        action_pl = K.placeholder(shape=(None, self.output_dim))  #onehot
        Q_pl = K.placeholder(shape=(None,1))
        V_pl = K.placeholder(shape=(None,1))
        
        #Find terms in bracket
        pi_vec = self.model.output
        pi = K.sum(pi_vec*action_pl,axis=1)    # get \pi(s_t, a_t) -- prob for specific action
        entropy = self.alpha*K.log(pi)
        temp = entropy - K.transpose(Q_pl) + K.transpose(V_pl)  #this is a row vec
        temp = K.transpose(temp)      #turn it into col vec
        
        #Find grad log(pi)
        pi_pl = self.model.output
        pars = self.model.trainable_weights
        grads = tf.gradients( K.log(pi_pl), pars, temp)   #scalar multiply by temp

        #Do learning
        # To get keras to apply updates given a custom gradients (i.e. run the above line) I had to alter the source
        # Code. It was easy to do. See line X in the get_updates function.
        opt = Adam(self.lr)
        loss = grads  #placeholder, I won't use it
        updates = opt.get_updates(loss = grads, params = pars, grads = grads)
         
        #This function will apply updates when called
        func = K.function(inputs = [state_pl, action_pl, Q_pl, V_pl], outputs = [], updates = updates)
        return func