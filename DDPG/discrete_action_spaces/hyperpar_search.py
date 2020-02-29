### I believe I have to really enhance the exploration
### at the start to get good results. These are controlled
### By batchsize, warm-up, and noise strength. So I will
### Vary these


import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent



def noisy_action(action):
    theta = 0.15
    sigma = 0.2
    mu = 0
    noisy_action =  theta * (mu - action) + sigma * np.random.randn()    
    return noisy_action


def train(par):
    """ There are other hyperparameters, but I'll just 
        look at these for now.
    """
    
    #Environment
    seed = 0
    env = gym.make('CartPole-v0')
    env.seed(seed)  # for comparison
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    #Agent
    gamma, lr, tau = par
    agent = Agent(num_states, num_actions, lr, gamma, seed_num = seed)
    agent.memory_size = 10**4
    agent.batchsize = 32
    learning_start = 2000
    agent.tau = tau


    #Train
    EPISODES = 500
    scores = []
    t1 = time.time()
    for e in range(1,EPISODES+1):
        state = env.reset()
        reward_sum = 0
        done = False
        steps = 0
        actions = []

        while not done:
            #env.render()
            state = np.reshape(state, [1, num_states])  #reshape for keras
            action_onehot = agent.act(state)
            action_scalar = np.dot(action_onehot,range(num_actions))
            actions.append(action_scalar)
            next_state, reward, done, _ = env.step(action_scalar)
            reward_sum += reward
            agent.remember(state[0], action_onehot, reward, next_state, done)
            state = next_state

            if len(agent.memory) > learning_start:
                agent.train_models()
                agent.actor.gumbel_temperature = max(0.999*agent.actor.gumbel_temperature, 0.1)
            steps += 1

        #Learn & print results
        scores.append(reward_sum)


    #agent.save_target_weights()
    plt.plot(scores)
    figname = 'gamma_' + str(gamma) + '_lr_' + str(lr) + '_tau_' + str(tau) + '.png'
    plt.title('gamma_' + str(gamma) + '_lr_' + str(lr) + '_tau_' + str(tau)) 
    plt.savefig('figs/' + figname)
    
    
    
################################### Main hyperparameter loop ##############################################################

gammas = [0.1,0.9]
lrs = [10**-i for i in [2,3,4]]
taus = [1.0,0.1,0.01,0.001]
pars = [(g,lr,tau) for g in gammas for lr in lrs for tau in taus]
print len(pars)

from multiprocessing import Pool
workers = Pool(6)
results = workers.map(train,pars)
workers.close()
workers.join()

#for result in results:
#    print result
    
np.savetxt('hyperpar_search.txt',results,fmt='%s')
