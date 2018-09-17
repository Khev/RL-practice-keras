import networkx as nx
import numpy as np



class Env:
    
    """ 
    Environment for DQN-Q-cab. At each step
    
    Input:
    G, osmnx graph
    state = 1 hot vector of initial position of agent
    
    Output:
    new_position, 1 hot vector
    reward, int    
    """
    
    def __init__(self, G, initial_state):
        self.G = G
        self.state = initial_state
        
        #Find action space -- a little tricky since nodes have different degrees
        #So I will have to outlaw some actions
        max_degree = max([len(G[node].values()) for node in G.nodes])
        self.num_actions = max_degree
        self.num_states = len(G.nodes)
        self.num_nodes = G.number_of_nodes()
        self.illegal_move_penalty = -1000
        self.pickup_reward = 10
        self.no_pickup_reward = -1
        
        #I will need the shortest paths
        self.shortest_paths = dict(nx.shortest_paths.all_pairs_shortest_path(G))
        
        
    def onehot_to_scalar(self,vector):
        return np.where(vector == 1)[0][0]
    
    def scalar_to_onehot_vector(self,scalar):
        return np.array([1 if i == scalar else 0 for i in range(self.num_states)])
    
    def scalar_to_onehot_tensor(self,scalar):
        return np.eye(self.num_states)[scalar:scalar+1]
        
    def step(self,action,cab):
        
        #If cab is serving, then continue along route (until finished)
        if cab.state == 'serving':
            next_state_scalar = cab.route[0]
            next_state_vector = self.scalar_to_onehot_vector(next_state_scalar)
            cab.route = cab.route[1:]
            if len(cab.route) == 0:
                cab.state = 'searching'
            cab.active_time += 1
            reward = 0       #I'll give zero reward, even though I won't let the cab learn when serving
            return next_state_vector, reward
        
        
        #If seaching, decide where to go
        else:
            #First 'correct' illegal actions, by setting their Q's to -inf, and picking a new random action
            state_scalar = self.onehot_to_scalar(self.state)
            neighbours = self.G[state_scalar].keys()
            if action > len(neighbours)-1:
                next_state_scalar = np.random.choice(range(len(neighbours)))
                next_state_vector = self.scalar_to_onehot_vector(next_state_scalar)
                reward = self.illegal_move_penalty
                cab.active_time += 1
                cab.idle_time += 1
                self.state = next_state_vector
                return next_state_vector, reward

            #And then proceed as normal
            next_state_scalar = neighbours[action]
            next_state_vector = self.scalar_to_onehot_vector(next_state_scalar)

            #Generate reward
            prob =  self.G.nodes[next_state_scalar]['prob']

            #If there is a trip
            if np.random.rand() < prob:
                destinations = list(self.G.nodes())
                destinations.pop(next_state_scalar)
                destination = np.random.choice(destinations)
                route = self.shortest_paths[next_state_scalar][destination]
                cab.route = route
                cab.state = 'serving'
                cab.active_time += 1
                reward = self.pickup_reward
                self.state = next_state_vector
                return next_state_vector, reward
        
            #If not, do X
            else:
                reward = self.no_pickup_reward
                cab.active_time += 1
                cab.idle_time += 1
                self.state = next_state_vector
                return next_state_vector, reward