import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical

#Sub classes
from actor import Actor
from critic import Critic


class Agent:
       
    def __init__(self,input_dim, output_dim, lr, gamma, tau, clipnorm, verbose):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        
        #Buffer for experience replay
        self.S = []
        self.A = []
        self.R = []
        self.S1 = []
        self.D = []
        self.memory_size = 10**3
        
        #Make actor and critic
        self.actor = Actor(input_dim,output_dim,lr, gamma, tau, clipnorm, verbose)
        self.critic = Critic(input_dim,output_dim, lr, gamma, tau, clipnorm, verbose)
        
           
    def learn(self):
            
        #get batch
        S,A,R,S1,D = self.get_batch()
            
        #train critic
        G = self.find_discounted_return(R)   #the return
        V, V1 = self.critic.model.predict(S), self.critic.model.predict(S1)
        self.critic.learn(S,R,V1,D,G)
        self.soft_update_target_network(self.critic)    
            
        #train actor
        #V, V1 = self.critic.model.predict(S), self.critic.model.predict(S1)
        self.actor.learn(S,R,V,V1)
        self.soft_update_target_network(self.critic)
        
        #Clear memory
        self.S, self.A, self.R, self.S1, self.D = [], [], [], [], []
        
        
    def find_discounted_return(self,rewards):
        rewards = rewards.flatten()
        R = np.zeros_like(rewards)
        rolling_sum = 0
        for t in reversed(range(len(R))):
            rolling_sum = rolling_sum*self.gamma + rewards[t]
            R[t] = rolling_sum
            
        #Normalize rewards
        R -= np.mean(R)
        R /= np.std(R)
            
        return np.array(R)

    
    def remember(self, state, action, reward, next_state, done):
        """ Add experience to buffer """
    
        self.S.append(state)   
        action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
        self.A.append(action_onehot)
        self.R.append([reward])
        self.S1.append(next_state)
        self.D.append([done*1.0])
            
            
    def get_batch(self):
        #indicies = np.random.choice(range(len(self.S)),self.batchsize)
        indicies = range(len(self.S))
        S = np.array(self.S)[indicies]
        A = np.array(self.A)[indicies]
        R = np.array(self.R)[indicies]
        S1 = np.array(self.S1)[indicies]
        D = np.array(self.D)[indicies]
        return S,A,R,S1,D
            
 
    def act(self, state):
        """ Choose action according to softmax """
        
        probs =  self.actor.model.predict(state)[0]
        action = np.random.choice(self.actions, p=probs)
        return action
    
    
    
    def make_tensor(self,vec):
        """ Turns a 1D array, x, into a 2d array / tensor =  [x]
            So that keras can read it
        """

        vec = np.reshape(vec, (1,len(vec)))
        return vec
    

            
    def save_target_weights(self):

        """ Saves the weights of the target 
            networks (only use the target during
            testing, so don't need to save the
            behavior network)
        """

        #Create directory if it doesn't exist
        dir_name = 'network_weights/'
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))

        #Now save the weights. I'm choosing ID by gamma, lr, tau
        pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau)  #save attached the extension

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = self.actor.target_model.get_weights()
        np.save(filename + pars_tag, actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = self.critic.target_model.get_weights()
        np.save(filename + pars_tag, critic_pars)




    def load_target_weights(self,gamma,lr,tau):

        """ Loads the weights of the target 
            networks, previously created using
            the save_target_wieghts() function
        """


        #Now save the weights. I'm choosing ID by gamma, lr, tau
        pars_tag = '_gamma_' + str(gamma) + '_lr_' + str(lr) + '_tau_' + str(tau) +'.npy'

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = np.load(filename + pars_tag)
        self.actor.target_model.set_weights(actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = np.load(filename + pars_tag)
        self.critic.target_model.set_weights(critic_pars)
        
        
        
    def soft_update_target_network(self,net):
        """
        Updates parameters of the target network according to the following
        where tau is a hyper parameter.
        
        theta_target = (1-tau)*theta_target + tau*theta_behavior
        
        Input: network = Actor or Critic class
        """
        
        pars_behavior = net.model.get_weights()       # these have form [W1, b1, W2, b2, ..], Wi = 
        pars_target = net.target_model.get_weights()  # bi = biases in layer i
        
        ctr = 0
        for par_behavior,par_target in zip(pars_behavior,pars_target):
            par_target = par_target*(1-self.tau) + par_behavior*self.tau
            pars_target[ctr] = par_target
            ctr += 1

        net.target_model.set_weights(pars_target)
