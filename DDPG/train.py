import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent


def train(par):
    """ There are other hyperparameters, but I'll just 
        look at these for now.
    """

    gamma,lr,tau = par

    #Environment
    env = gym.make('MountainCarContinuous-v0')
    env.seed(1)  # for comparison
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    #Agent
    agent = Agent(num_states, num_actions, lr, gamma)
    agent.memory_size = 10**4
    agent.batchsize = 256
    learning_start = 25*agent.batchsize
    agent.tau = tau
    
    #Train
    EPISODES = 200
    MAX_STEPS = 1000
    scores = []
    t1 = time.time()
    for e in range(1,EPISODES+1):
        state = env.reset()
        reward_sum = 0
        done = False
        steps = 0

        while not done:
            # env.render()
            state = np.reshape(state, [1, num_states])  #reshape for keras
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.remember(state[0], action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > learning_start:
                agent.train_models()

            steps += 1
            if done or steps > MAX_STEPS:
                break

        #Learn & print results
        scores.append(reward_sum)
        
    #Print results
    #I'll assess performace as the mean of the last 100 episodes
    cutoff = EPISODES / 2
    string =  '(gamma, lr, tau, score) = ' + str((gamma, lr, tau, np.mean(scores[cutoff:])))
    t2 = time.time()
    print string
    print 'took ' + str( (t2-t1)/60.0 ) + ' mins \n'
    
    #save figure
    plt.figure(figsize=(9,6))
    plt.plot(scores)
    plt.xlabel('episode', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.title('(gamma,lr,tau) = ' + str((gamma,lr,tau)))
    filename = 'stats/gamma_' + str(gamma) + '_lr_' + str(lr) + '_tau_' + str(tau) +  '.png'
    plt.savefig(filename)
   

    return string
    
################################### Main hyperparameter loop ##############################################################


#par = (0.1, 0.001, 0.01)
#train(par)


gammas = [0.1,0.99]
lrs = [0.0001, 0.001, 0.01]
taus = [0.001, 0.01, 0.1]
pars = [(g,lr,tau) for g in gammas for lr in lrs for tau in taus ]

print len(pars)

from multiprocessing import Pool
workers = Pool(6)
results = workers.map(train,pars)
np.savetxt('stats/hyperpar_results.txt',results,fmt="%s")

