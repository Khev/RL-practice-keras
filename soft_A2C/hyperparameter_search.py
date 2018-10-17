### I believe I have to really enhance the exploration
### at the start to get good results. These are controlled
### By batchsize, warm-up, and noise strength. So I will
### Vary these


import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
import gym,time


def train(pars):
    """ There are other hyperparameters, but I'll just 
        look at these for now.
    """
    
    alpha,tau, batchsize = pars

    #Environment
    env = gym.make('CartPole-v0')
    env.seed(0)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n


    #Agent
    lr,gamma = 3*10**-4, 0.99
    clipnorm, verbose = False, False
    agent = Agent(input_dim, output_dim, lr, gamma, tau, alpha, clipnorm, verbose)
    agent.memory_size = batchsize
    agent.batchsize = batchsize

    #Train
    EPISODES = 10**4
    scores = []
    t1 = time.time()
    for e in range(1,EPISODES+1):
        state = env.reset()
        state = agent.make_tensor(state)
        reward_sum = 0
        done = False
        while not done:

            #Do main step
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = agent.make_tensor(next_state)
            agent.remember(state[0],action,reward,next_state[0],done) #want to remember state as a vec
            state = next_state
            if e >= 2:
                agent.learn()

        #Print results
        scores.append(reward_sum)
 
    plt.figure()
    string = 'alpha_'+str(alpha)+'_tau_'+str(tau)+'_batchsize_'+str(batchsize)
    plt.title(string)
    plt.plot(scores,alpha=0.5)
    plt.plot(agent.window_average(scores,100),'r-')
    plt.savefig('figs/' + string + '.png')
    t2 = time.time()
    print 'took ' + str( (t2-t1) / 60.0 / 60.0) + ' hours'
    return 
    
    
################################### Main hyperparameter loop ##############################################################

alphas = [0.1,0.01,0.001,0.0001]
taus = [0.1,0.01,0.001,0.0001]
bs = [5]
pars = [(a,t,b) for a in alphas for t in taus for b in bs]
print len(pars)

from multiprocessing import Pool
workers = Pool(6)
results = workers.map(train,pars)
workers.close()
workers.join()