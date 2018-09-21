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


def train():
    """ There are other hyperparameters, but I'll just 
        look at these for now.
    """
    
    #Environment
    env = gym.make('MountainCarContinuous-v0')
    env.seed(1)  # for comparison
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    #Agent
    lr = 0.01
    gamma = 0.99
    agent = Agent(num_states, num_actions, lr, gamma)
    agent.memory_size = 10**4
    agent.batchsize = 256
    learning_start = 3*agent.batchsize
    agent.tau = 0.001
    
    #Train
    EPISODES = 30
    MAX_STEPS = 500
    scores = []
    t1 = time.time()
    for e in range(1,EPISODES+1):
        state = env.reset()
        reward_sum = 0
        done = False
        steps = 0
        x_max, x_min = state[0], state[0]   #state = [position, velocity]

        while not done:
            # env.render()
            #Noisy actions
            state = np.reshape(state, [1, num_states])  #reshape for keras
            action = agent.act(state)
            p = (1.0*e) / EPISODES
            action = p*noisy_action(action) + (1-p)*action   #make action space noisy
           
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.remember(state[0], action, reward, next_state, done)
            state = next_state
            
            x = state[0]  #position
            x_max = max(x,x_max)
            x_min = min(x,x_min)

            if len(agent.memory) > learning_start:
                agent.train_models()

            steps += 1
            if done or steps > MAX_STEPS:
                break

        #Learn & print results
        scores.append(reward_sum)
        t2 = time.time()
        x_max, x_min = np.round(x_max,2), np.round(x_min, 2)
        if e % 1 == 0:
            print '(episode, score, steps, T (mins)) = ' +str((e,int(reward_sum),steps,int((t2-t1)/60.0)))
            print("x_max = %.2f" % round(x_max,2))
            print("x_min = %.2f" % round(x_min,2))
            print("amplitude = %.2f" % round(x_max-x_min,2))
            print '\n'

        
    #Print results
    #I'll assess performace as the mean of the last 100 episodes
    cutoff = EPISODES / 2
    print '(gamma, lr, tau, score) = ' + str((gamma, lr, tau, np.mean(scores[cutoff:])))
    t2 = time.time()
    print 'took ' + str( (t2-t1)/60.0 ) + ' mins \n'
    
    
    
################################### Main hyperparameter loop ##############################################################

train()
