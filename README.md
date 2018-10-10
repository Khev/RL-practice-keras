# Deep Reinforcement Learning

Here I am implementing various RL algorithms, using python 2.7.  I will use keras for the neurals nets. I'm going to
use the OpenAI gym to test the algorithms. I list the methods below, which roughly divide into two
categories.

I took / adjusted code from various online sources, which I inexhaustively list below (and in codebase
itself).

### Value based methods

- [x] Q-learning (tabular)
- [x] Deep Q-Network (DQN)
- [x] Dueling DQN  
- [ ] DQN with prioritised replay
- [ ] Double Deep Q-network (DDQN)  
- [ ] Distributional bellman

### Policy based methods

- [x] Policy gradient -- REINFORCE & with baseline.
- [x] Actor critic (A2C)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [ ] Proximal policy optimization (PPO)


### Multi-agent

- [] Muti-agent deep deterministic policy gradient (MADDPG) 
- [] Actor-Attention-Critic (AAC)

##  Resources


### Papers
- [Q-learning]()
- [DQN](https://www.nature.com/articles/nature14236)
- [Dueling DQN](://arxiv.org/abs/1511.06581)
- [DQN with prioritized replay](https://arxiv.org/abs/1511.05952)
- [Distriubtional Bellman](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
- [DDPG](http://proceedings.mlr.press/v32/silver14.pdf)
- [MADDPG](https://arxiv.org/abs/1706.02275)
- [AAC](https://arxiv.org/abs/1810.02912)

### Blogs
- [Arthur
  Juliana](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- [yanpanlau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)


### Textbooks
- [Sutton](http://incompleteideas.net/book/the-book-2nd.html)


## Acknowledgements

 - [@germain-hug](https://github.com/germain-hug/Deep-RL-Keras)
 - [@Keras-RL](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)
 - [@keon](https://github.com/keon/policy-gradient/blob/master/pg.py)
