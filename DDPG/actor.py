import numpy as np
import tensorflow as tf
from keras.initializers import RandomUniform
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten
from keras import backend as K
from keras.optimizers import Adam 



class Actor:
    
    """ Actor for DDPG """
    
    def __init__(self,input_dim, output_dim,lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr                                            # learning rate for optimizer
        self.act_range = 1.0     #change this 
        self.model = self._make_network()
        self.target_model = self._make_network()                # target networks to stabilize learning.
        self.target_model.set_weights(self.model.get_weights()) # clone the networks
        self.adam_optimizer = self.optimizer()

       
    #Taken from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
    def _make_network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape = (self.input_dim,))
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #x = Flatten()(x)   # I assume this is if the input is a convolutional neural net?
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        out = Dense(self.output_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.act_range)(out)
        return Model(inp, out)
    
    
    def learn(self,states,grad_actions):
        self.adam_optimizer([states,grad_actions]) 
    
    
    def optimizer(self):
        """ Loss = grad_(theta) \mu(s)  grad_(actions) Q(s,a)
        
            where
            s = state
            a = action
            \mu_(\theta)(s) = the deterministic policy (actor)
            Q(s,a) = the Q-value, from the critic
        """
        
        # taken from https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
        # I believe this is a work around to get keras to learn **given a gradient**
        # As opposed to bunch of x_train, y_trains?
        
        #Inputs
        state_pl = self.model.input
        action_grads_pl = K.placeholder(shape=(None,1))  
                                        
        #Find grad_(pars) mu(state)
        mu_pl = self.model.output
        pars = self.model.trainable_weights
        pars_grad_mu = tf.gradients(mu_pl, pars, -action_grads_pl)
        
        #grads_and_pars = zip(pars_grad_mu, pars)  #keras needs this form
        #updates = tf.train.AdamOptimizer(self.lr).apply_gradients(grads_and_pars)

        # The gradients as defined above work on my mac, but not ubuntu.
        # Below I am trying a workaround. I changed the keras source code 
        # To get this working. Specifically, I make the optimizer.get_updates()
        # function accept custom gradients. It was easy to do.
        
        opt = Adam(self.lr)
        loss = pars_grad_mu  #placeholder, I won't use it
        updates = opt.get_updates(loss = loss, params = pars, grads = pars_grad_mu)

        return K.function(inputs = [state_pl, action_grads_pl], outputs = [], updates = updates)
        #return K.function(inputs = [state_pl, action_grads_pl], outputs = [updates])
      