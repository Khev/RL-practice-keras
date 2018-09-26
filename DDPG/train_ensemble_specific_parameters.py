# Here I'm running an ensemble of trainings at a specific parameter instance.
# I want to see how the learning stability varies
# I will seed each run by its trial number, for reproducibility 


import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent


def train(seed):

    gamma,lr,tau = 0.99, 0.0001, 0.001

    #Environment
    env = gym.make('MountainCarContinuous-v0')
    env.seed(1)  # for comparison
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    #Agent
    agent = Agent(num_states, num_actions, lr, gamma, seed_num = seed)
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
    string =  '(gamma, lr, tau, seed,score) = ' + str((gamma, lr, tau, seed, np.mean(scores[cutoff:])))
    t2 = time.time()
    print string
    print 'took ' + str( (t2-t1)/60.0 ) + ' mins \n'
    
    #save figure
    plt.figure(figsize=(9,6))
    plt.plot(scores)
    plt.xlabel('episode', fontsize=18)
    plt.ylabel('score', fontsize=18)
    plt.title('(gamma,lr,tau) = ' + str((gamma,lr,tau)))
    filename = 'stats/gamma_' + str(gamma) + '_lr_' + str(lr) + '_tau_' + str(tau)
    filename = filename + '_trial_' + str(seed) + '.png'
    plt.savefig(filename)
   
    #Save the target_networks
    agent.save_target_weights()

    return string
    
################################### Main loop ##############################################################



num_trials = 15
seeds = range(num_trials)

from multiprocessing import Pool
workers = Pool(5)
results = workers.map(train,seeds)
np.savetxt('stats/ensemble_specific_parameters.txt',results,fmt="%s")

