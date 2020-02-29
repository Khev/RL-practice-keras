import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Input, concatenate


class CriticQ:
    
    """ Q-funtion for critic """
        
    def __init__(self,input_dim,output_dim,lr,gamma,tau,alpha,clipnorm, clipnorm_val,verbose = False):
        
        #Pars
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.verbose = verbose
        self.clipnorm = clipnorm
        self.clipnorm_val = clipnorm_val
        
        #Make models
        self.model = self._make_network()
        self.target_model = self._make_network()                       #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())         #clone the network
        
        #Optimizer
        self.opt = self.optimizer()

        
    def _make_network(self):
        """ Q(s,a) -- function of two arguments """
        
        S = Input(shape=(self.input_dim,))
        A = Input(shape=(self.output_dim,))
        x = concatenate([S,A])
        x = Dense(256, activation = 'relu')(x)
        x = Dense(256, activation = 'relu')(x)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = [S,A], outputs = out)
        return model
    
    
    def learn(self,S,A,R,V_target):        
        """ 
            S = batch of states
            A = batch of actions
            R = batch of Q-vals
            V_targets = batch of values, from target network
        """
            
        self.opt([S,A,R,V_target])
        
        
        
    def optimizer(self):
        """ 
           The gradient of the loss function L is
           
           \grad L = \grad_pars Q (  Q(s_t, a_t) - r(s_t, a_t) - gamma* V_target(s_{t+1}) )
           
           where,
           r = reward
           gamma = discount factor
           V_target = Value target network
           s_{t+1} = next state
           
        """
        
        #Input
        S_pl, A_pl = self.model.input
        Q_pl = self.model.output

        #Find term in bracket
        V_target_pl = K.placeholder(shape=(None,1))
        R_pl = K.placeholder(shape=(None,1)) 
        temp = Q_pl - R_pl - self.gamma*V_target_pl

        #Find gradient
        pars = self.model.trainable_weights
        grads = tf.gradients(Q_pl,pars,-temp)  #scalar multiply by temp
        
        #Clip gradients
        if self.clipnorm == True:
            grads = tf.clip_by_global_norm(grads,self.clipnorm_val)[0]            

        #Do learning
        #To get keras to apply updates given a custom gradients (i.e. run the above line) I had to alter the source
        #Code. It was easy to do. See line X in the get_updates function.
        opt = Adam(self.lr)
        loss = grads
        updates = opt.get_updates(loss = loss, params = pars, grads = grads)

        #This function will apply updates when called
        func = K.function(inputs=[S_pl,A_pl,R_pl,V_target_pl],outputs=[], updates = updates)
        return func