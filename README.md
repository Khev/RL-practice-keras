# Deep Reinforcement Learning

Here I am implementing various RL algorithms, using python 2.7.  I will use keras for the neurals nets. I'm going to
use the OpenAI gym to test the algorithms. I list the methods below, which roughly divide into two
categories.

I took / adjusted code from various online sources, which I inexhaustively list below (and in the code
itself).

### Value based methods

- [x] Q-learning (tabular)
- [x] Deep Q-Network (DQN)
- [x] Double DQN  (DDQN) 
- [ ] DQN with prioritised replay
- [ ] Distributional bellman


### Policy based methods

- [x] Policy gradient -- REINFORCE & with baseline.
- [x] Actor critic (A2C)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [x] Proximal policy optimization (PPO)
- [ ] Soft Actor-Critic (soft AC)

### Multi-agent

- [ ] Muti-agent deep deterministic policy gradient (MADDPG) 
- [ ] Actor-Attention-Critic (AAC)
- [ ] Value Decompostion Networks (VDN) 
- [ ] QMIX

### Others

- [ ] Explore-and-go
- [ ] Curiosity driven learning (CDL)
- [ ] Rainbow (RB) 

##  Resources


### Papers
- [Q-learning]()
- [DQN](https://www.nature.com/articles/nature14236)
- [Dueling DQN](http://arxiv.org/abs/1511.06581)
- [DQN with prioritized replay](https://arxiv.org/abs/1511.05952)
- [Distriubtional Bellman](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
- [PPO](http://arxiv.org/abs/1707.06347)
- [Soft AC](https://arxiv.org/pdf/1801.01290.pdf)
- [DDPG](https://arxiv.org/abs/1509.02971)
- [MADDPG](https://arxiv.org/abs/1706.02275)
- [AAC](https://arxiv.org/abs/1810.02912)
- [CDL](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
- [RB](https://arxiv.org/abs/1710.02298) 
- [EG](https://eng.uber.com/go-explore/)
- [MARL review article](https://arxiv.org/pdf/1810.05587v1.pdf) 
- [QMIX](https://arxiv.org/abs/1803.11485)
- [VDN](https://arxiv.org/pdf/1706.05296.pdf)

### Blogs
- [Arthur
  Juliana](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
- [yanpanlau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
- [Gumble softmax trick](http://amid.fish/humble-gumbel)
- [review_blog](https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3)
- [TRPO intro](https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9)

### Textbooks
- [Sutton](http://incompleteideas.net/book/the-book-2nd.html)


## Acknowledgements

 - [@germain-hug](https://github.com/germain-hug/Deep-RL-Keras)
 - [@Keras-RL](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)
 - [@keon](https://github.com/keon/policy-gradient/blob/master/pg.py)
