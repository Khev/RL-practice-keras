import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Input, concatenate


class CriticV:
    
    """ Value function of critic """

    def __init__(self,input_dim,output_dim,lr,gamma,tau,alpha,clipnorm, clipnorm_val,verbose = False):
        
        #Pars
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr  #learning rate for optimizer
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha       #entropy parameter
        self.verbose = verbose
        self.clipnorm = clipnorm
        self.clipnorm_val = clipnorm_val

        
        #Make models
        self.model = self._make_network()
        self.target_model = self._make_network()                       #we have target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights())         #clone the network
        
        #Make the optimizers
        self.opt = self.optimizer()

        
    def _make_network(self):
        """ V(s) """
        
        S = Input(shape=(self.input_dim,))
        x = Dense(256, activation = 'relu')(S)
        x = Dense(256, activation = 'relu')(x)
        out = Dense(1, activation = 'linear')(x)
        model = Model(inputs = S, outputs = out)
        return model
    
    
    def learn(self,S,Q,Pi):
        self.opt([S,Q,Pi])
    
 
    def optimizer(self):
        
        """ 
           The gradient of the loss function L is
           
           \grad L = \grad_pars V (  V(s_t) - Q(s_t, a_t) + \alpha*log( \pi(s_t, a_t) )  )
          
        """
      
    
        #Find terms in bracket
        S_pl = self.model.input
        Pi_pl = K.placeholder(shape=(None,1))
        Q_pl = K.placeholder(shape=(None,1))
        V_pl = self.model.output
        temp = V_pl - Q_pl + self.alpha*K.log(Pi_pl)

        #Find gradient
        pars = self.model.trainable_weights
        grads = tf.gradients(V_pl, pars, -temp)  #scalar multiply by temp
        
        #Clip gradients
        if self.clipnorm == True:
            grads = tf.clip_by_global_norm(grads,self.clipnorm_val)[0]            
        
        #Do learning
        #To get keras to apply updates given a custom gradients (i.e. run the above line) I had to alter the source
        #Code. It was easy to do. See line X in the get_updates function.
        opt = Adam(self.lr)
        loss = grads  #placeholder, keras doesn't use it
        updates = opt.get_updates(loss = loss, params = pars, grads = grads)

        #This function will apply updates when called
        func = K.function(inputs = [S_pl, Q_pl, Pi_pl], outputs = [], updates = updates)
        return func