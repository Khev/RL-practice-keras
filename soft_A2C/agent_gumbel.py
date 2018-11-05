import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical

#Sub classes
from actor_gumbel import Actor
from criticQ import CriticQ
from criticV import CriticV


class Agent:
       
    def __init__(self,input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        #Buffer for experience replay
        self.S = []
        self.A = []
        self.R = []
        self.S1 = []   #next state
        self.D = []
        self.memory_size = 2000
        self.batchsize = 32 
        
        #Make actor and critic
        self.actor = Actor(input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose)
        self.criticQ = CriticQ(input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose)
        self.criticV = CriticV(input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose)
        
           
    def learn(self):
            
        #get batch
        S,A,R,S1,D = self.get_batch()
        
        
        #Train actor --- the actions are evaluted from the actor
        A_temp = self.actor.predict(S)
        
        self.actor.learn(S,A,Q,V_target)
        self.soft_update_target_network(self.actor)
        
        #Get what we need for learning
        Q = self.criticQ.model.predict([S,A])
        V_target = self.criticV.target_model.predict(S1)  #value of NEXT state
        
        Pi_vec = self.actor.model.predict(S)
        Pi = np.sum(Pi_vec*A,axis=1)          #now a batch of \pi(s_t,a_t)
        Pi = np.array([[i] for i in Pi])      #reshape to np.array([[x1],[x2],[x3]]), same as Q, V_target.
  
        #train actor
        self.actor.learn(S,A,Q,V_target)
        self.soft_update_target_network(self.actor)
        
        #train critic
        self.criticQ.learn(S,A,R,V_target)
        self.soft_update_target_network(self.criticQ)
        
        #train value funtion
        #Here, the actions are sampled from the current policy
        Pi_vec = self.actor.model.predict(S)
        Pi = np.sum(Pi_vec*A,axis=1)          #now a batch of \pi(s_t,a_t)
        Pi = np.array([[i] for i in Pi])      #reshape to np.array([[x1],[x2],[x3]]), same as Q, V_target.
        
        self.criticV.learn(S,Q,Pi)
        self.soft_update_target_network(self.criticV)

    
    def remember(self, state, action, reward, next_state, done):
        """ Add experience to buffer """
    
        if len(self.S) <= self.memory_size:
            self.S.append(state)   
            action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
            self.A.append(action_onehot)
            self.R.append([reward])
            self.S1.append(next_state)
            self.D.append(done)
            
        else:
            self.S[0] = state   
            action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
            self.A[0] = action_onehot
            self.R[0] = [reward]
            self.S1[0] = next_state
            self.D[0] = done
            
            
    def get_batch(self):
        indicies = np.random.choice(range(len(self.S)),self.batchsize)
        S = np.array(self.S)[indicies]
        A = np.array(self.A)[indicies]
        R = np.array(self.R)[indicies]
        S1 = np.array(self.S1)[indicies]
        D = np.array(self.D)[indicies]
        return S,A,R,S1,D
            
 
    def act(self, state):
        """ Choose action according to softmax """
        
        #The softmax gumbel trick outputs an almost 1-hot vector (i.e elements sum to one, with way way bigger than others)
        #I need to turn this into a 'hard' onehot vector
        
        action_soft_onehot =  self.actor.model.predict(state)[0]
        action_index = np.argmax(action_soft_onehot)
        action_hard_onehot = np.array([1 if i == action_index else 0 for i in range(len(action_soft_onehot))])
        return action_hard_onehot

    
    
    def make_tensor(self,vec):
        """ Turns a 1D array, x, into a 2d array / tensor =  [x]
            So that keras can read it
        """

        vec = np.reshape(vec, (1,len(vec)))
        return vec
    
    
    
    def window_average(self,x,N):
        """ Does a window average of size
            N on the array x
        """
        return np.convolve(x, np.ones((N,))/N, mode='valid')
    

            
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
