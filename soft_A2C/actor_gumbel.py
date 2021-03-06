import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.initializers import RandomUniform
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
        
        #Hyperpars chosen according to the paper (see README)
        S = Input(shape=(self.input_dim,))
        x = Dense(256, activation = 'relu')(S)
        x = Dense(256, activation = 'relu')(x)
        
        # Now do the softmax gumbel trick: (which outputs a one-hot vector)
        # Apply softmax to (g_i + logits) / temperate
        # where g_i is gumbel noise, and temperature is a 
        # softness par (when small, almost exactly a one-hot vec)
        logits = Dense(self.output_dim, kernel_initializer=RandomUniform())(x)
        z = Lambda(self.GumbelNoise)(logits)    #add noise
        z = Lambda(lambda x: x / self.gumbel_temperature)(z) #divide by temperature
        out = Dense(self.output_dim, activation='softmax')(z)  #then softmax
        return  Model(S, out)
    
    
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
        S_pl = self.model.input
        A_pl = K.placeholder(shape=(None, self.output_dim))  #onehot
        Q_pl = K.placeholder(shape=(None,1))
        V_pl = K.placeholder(shape=(None,1))
        
        #Find terms in bracket
        pi_vec = self.model.output
        pi = K.sum(pi_vec*A_pl,axis=1)    # get \pi(s_t, a_t) -- prob for specific action
        entropy = self.alpha*K.log(pi)
        temp = entropy - K.transpose(Q_pl) + K.transpose(V_pl)  #this is a row vec
        temp = K.transpose(temp)      #turn it into col vec
        
        #Find grad log(pi)
        pi_pl = self.model.output
        pars = self.model.trainable_weights
        grads = tf.gradients( K.log(pi_pl), pars, temp)   #scalar multiply by temp

        #Do learning
        #To get keras to apply updates given a custom gradients (i.e. run the above line) I had to alter the source
        #Code. It was easy to do. See line X in the get_updates function.
        opt = Adam(self.lr)
        loss = grads  #placeholder, I won't use it
        updates = opt.get_updates(loss = grads, params = pars, grads = grads)
         
        #This function will apply updates when called
        func = K.function(inputs = [S_pl, A_pl, Q_pl, V_pl], outputs = [], updates = updates)
        return func
    
    
    
    def GumbelNoise(self,logits):
        
        """ Adds gumbels noise to the logits
            I generate the gumbel noise by 
            applying the inverse CDF to uniform
            noise. 

            The inverse CDF of the gumbel is
            -log( -log(x) )

        """
        U = K.random_uniform(K.shape(logits), 0, 1)
        y = logits - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
        return y