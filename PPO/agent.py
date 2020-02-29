import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical

#Sub classes
from actor import Actor
from critic import Critic


class Agent:
    
    """ PPO agent. Code inspired from https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py """
       
    def __init__(self,input_dim, output_dim, lr, gamma, loss_clipping, c1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions = range(output_dim)  
        self.lr = lr
        self.gamma = gamma
        self.loss_clipping = loss_clipping  #for actor loss function
        self.c1 = c1   #weight for entropy term in actor loss function
        self.num_epochs = 10
        self.batchsize = 10
        
        #These will store the samples from which the agent will learn
        self.states = []
        self.actions = []
        self.pi_vecs = []
        self.rewards = []
        
        #Make actor and critic
        self.actor = Actor(input_dim,output_dim,lr,gamma,loss_clipping,c1)
        self.critic = Critic(input_dim,output_dim, self.lr)
        
    
    def get_batch(self):
        """ For now, just take all the thing in memory """
        
        #Turn lists into arrays
        S = np.array(self.states)             #stack of state vectors
        A = np.array(self.actions)     # stack of one-hot action vectors
        Pi = np.array(self.pi_vecs)           # stack of pi_vec, where pi_vec_i = \pi(s_i, a_i)
        R = np.array(self.rewards)            # stack of rewards vecs (a vector with 1 element = scalar)
        
        return S,A,Pi,R
    
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.pi_vecs = []
        self.rewards = []
    
    
    def train_models(self):
    
        #Easist implementation of batching, will make more sophisticated later
        states, actions, pi_vecs, rewards = self.get_batch()
        self.clear_memory() 
        
        #Compute inputs for the optimizers
        discounted_return = self.find_discounted_return(rewards)
        values = self.critic.model.predict(states)
        values.resize(len(values))
        advantages = discounted_return - values
        
        #Do the training
        old_predictions = pi_vecs
        actor_loss = self.actor.model.fit([states, advantages, old_predictions], [actions], \
                      batch_size=self.batchsize, shuffle=True, epochs=self.batchsize, verbose=False)
        critic_loss = self.critic.model.fit([states], [discounted_return], \
                      batch_size=self.batchsize, shuffle=True, epochs=self.batchsize, verbose=False)
        
        #Soft update target networks -- none for PPO
        #self.soft_update_target_network(self.actor)
        #self.soft_update_target_network(self.critic)
        
        
    def find_discounted_return(self,rewards):
        R = np.zeros_like(rewards)
        rolling_sum = 0
        for t in reversed(range(len(R))):
            rolling_sum = rolling_sum*self.gamma + rewards[t]
            R[t] = rolling_sum
            
        #Normalize rewards
        R -= np.mean(R)
        R /= np.std(R)
            
        return np.array(R)
    
        
    
    def remember(self, state, action, pi_vec, reward):
        self.states.append(state)
        
        action_onehot = to_categorical(action,self.output_dim) #optimizers use one-hot
        self.actions.append(action_onehot)
        
        self.pi_vecs.append(pi_vec)
        self.rewards.append(reward)
        

    def act(self, state):
        """ Choose action according to softmax """
        
        
        #Recall the actor model takes three inputs: [states, advantages, pi_old]
        #The latter two for the loss function fitting
        #For selecting an action, I only need the first
        #So, I will use a dummy variables
        
        #Prepare dummy vars
        num_samples = state.shape[0]
        adv_dummy = np.zeros((num_samples, 1))
        pi_old_dummy = np.zeros((num_samples, self.output_dim))
        inp = [state, adv_dummy, pi_old_dummy]            
                                
        #Find action
        pi_vec = self.actor.model.predict(inp)[0]      # prob of actions 
        action = np.random.choice(range(self.output_dim), p=pi_vec)
        return action, pi_vec
    
    
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
        
        
    def save_target_weights(self):

        """ Saves the weights of the target 
            network (only use the target during
            testing, so don't need to save tje
            behavior)
        """

        #Create directory if it doesn't exist
        dir_name = 'network_weights/'
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))

        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau)
        else:
            pars_tag = '_gamma_' + str(self.gamma) + '_lr_' + str(self.lr) + '_tau_' + str(self.tau) \
            +'_seed_' + str(self.seed_num)
            

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
            network, previously created using
            the save_target_wieghts() function
        """


        #Now save the weights. I'm choosing ID by gamma, lr, tau
        if self.seed_num == False:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_' + str(self.tau) + '.npy'
        else:
            pars_tag = '_gamma_' + str(self.gamma)+'_lr_'+str(self.lr)+'_tau_'+str(self.tau)+'_seed_' \
            +str(self.seed_num)+ '.npy'

        #Actor target network
        filename = 'network_weights/actor_target'
        actor_pars = np.load(filename + pars_tag)
        self.actor.target_model.set_weights(actor_pars)

        #Critic target network
        filename = 'network_weights/critic_target'
        critic_pars = np.load(filename + pars_tag)
        self.critic.target_model.set_weights(critic_pars)