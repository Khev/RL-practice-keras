####

# Environment for the real world nyc graph. Also, some auxiliary functions for working on this environment.

####

import networkx as nx
import numpy as np
import osmnx as ox
import datetime
import copy
import csv 
from keras.utils import to_categorical

class Env:
    
    """ 
    A taxi moves on a street network. His goal is to minimize his idle time
    (fraction of time empty). At each step, available passenger trips are loaded
    at nodes on the graph If the cab is at one of these, he gets a reward, and 
    then goes to the desinations. If not, then he moves somewhere 
    
    Input:
    G, osmnx graph
    trips = np.array([pickup time (int), pickup_place(int), dropoff time (int), dropoff place (int) ])
    position, int
    
    Output:
    new_position, int
    reward, float 
    """
    
    def __init__(self, G, trips, state):
        self.trips = trips
        self.G = G
        self.state = state
        self.active_time = 0
        self.available_trips = []
        self.idle_time = 0
        self.delta = 12    #time trips wait before disappearing
        self.illegal_move_penalty = -1000
        self.pickup_reward = 100
        self.no_pickup_reward = -1
        self._make_converter()
        self.num_nodes = G.number_of_nodes()
        
        #Find action space -- a little tricky since nodes have different degrees
        #So I will have to outlaw some actions
        max_degree = max([len(G[node].values()) for node in G.nodes])
        self.num_actions = max_degree
        self.num_states = G.number_of_nodes()
        
        
    def find_tau(self):
        return (1.0*self.idle_time) / self.active_time
                
        
    def _make_converter(self):
        """ The states of this enviroments are the nodes of the street network.
            These nodes are labeled by their osmnx IDs. I want to express the 
            states of the graph as one-hot vectors, so I pass them through a neural net. 
            
            So, this does that
        """
    
    
        nodes_osids = list(self.G.nodes)
        nodes_int = range(len(nodes_osids))
        self.converter = {}
        
        for label_int, label_osmnx in enumerate(nodes_osids):
            self.converter[label_osmnx] = label_int 
        
        
    def convert(self,node_osmnx):
        """ 
        Does the coversion -- see above
        
        Input: int, osmnx ID
        Output: one hot vector
        """
        
        state_scalar = self.converter[node_osmnx]  #first get as an int -- states are labeled by their osmnx id
        state_vector = to_categorical(state_scalar,self.num_states)  #one-hot
        return state_vector
        
        
    def reset(self,state,trips):   #note, not resetting to original position
        self.trips = trips
        self.active_time = 0
        self.available_trips = []
        self.idle_time = 0
        self.state = state
        
                
    def step(self,action):
        
        #Load new trips
        for trip in self.trips:
            if trip[0] <= self.active_time: 
                self.available_trips.append(list(trip))
                self.trips = self.trips[1:]   #delete trip after this timestep
            else:
                break
        
        #Move to next state
        neighbours = self.G[self.state].keys()
        
        #First check if action is legal
        if action > len(neighbours)-1:
            action = np.random.choice(range(len(neighbours)))
            reward = self.illegal_move_penalty   #penalize the illegality
            next_state = neighbours[action]
            self.state = next_state
            self.active_time += 1
            self.idle_time += 1
            
            #Remove trips
            self.available_trips = [trip for trip in self.available_trips if \
                                    (self.active_time-trip[0])<=self.delta]
            return self.state, reward
        
        #Else move, and proceed as normal
        next_state = neighbours[action]
        self.state = next_state
        
        #Generate rewards
        found_trip = False
        for trip in self.available_trips:
            if self.state == trip[1]:
                destination = trip[3]
                duration = trip[2] - trip[0]
                found_trip = True
                break
                
        if found_trip == True:  #if you're on a hot spot
            reward = self.pickup_reward   #get reward
            self.active_time += duration
            self.state = destination      #move to directly to destiantion -- can do this when I only have 1 taxi
        else:
            reward = self.no_pickup_reward
            self.idle_time += 1
            self.active_time += 1
            
        #Remove trips that are no longer active
        self.available_trips = [trip for trip in self.available_trips if (self.active_time-trip[0])<=self.delta]
            
        return self.state, reward
    
    
    # I parameterized the actions for the Qcab differently here; action \in {Nodes},
    # whereas for the Qcab, action = 0,1,2,  standing for the first,second,third, neighbour
    def step_modelcab(self,action):
        #Move to next state
        next_state = action
        self.state = next_state

        #Get reward -- first load trips
        for trip in self.trips:
            if trip[0] <= self.active_time:
                self.available_trips.append(list(trip))
                self.trips = self.trips[1:]   #delete trip after this timestep
            else:
                break
        
        found_trip = False
        for trip in self.available_trips:
            if self.state == trip[1]:
                destination = trip[3]
                duration = trip[2] - trip[0]
                found_trip = True
                break
                
        if found_trip == True:  #if you're on a hot spot
            reward = 1
            nodes = list(self.G.nodes())
            nodes.remove(self.state)
            self.active_time += duration
            self.state = destination
        else:
            reward = 0
            self.idle_time += 1
            self.active_time += 1
            
        #Remove trips that are no longer active
        self.available_trips = [trip for trip in self.available_trips if (self.active_time-trip[0])<=self.delta]
            
        return self.state, reward

    
    
    
##############################################################################################################################
##############################################################################################################################
                                       # Auxiliary functions
    
    
    
def get_subgraph():
    """Grabs a sub-graph of NYC
    
       Output:
       G = osmnx graph, street network
    
    """
    G = ox.graph_from_point((40.758896,-73.985130), distance=1500, network_type='drive')
    Gs = [i for i in nx.strongly_connected_component_subgraphs(G)]
    max_index = np.argmax([G2.number_of_nodes() for G2 in Gs])
    G = Gs[max_index]
    return G




def get_full_nyc_graph():
    """Grabs a sub-graph of NYC
    
       Output:
       G = osmnx graph, street network
    
    """
    G = ox.graph_from_place('Manhattan',network_type='drive')
    Gs = [i for i in nx.strongly_connected_component_subgraphs(G)]
    max_index = np.argmax([G2.number_of_nodes() for G2 in Gs])
    G = Gs[max_index]
    return G




def get_tripdata(G, start_day, end_day):
    """
    
    Input:
    G = osmnx graph, streetnetwork
    start_day = int, 0 = 1st of Jan, 1 = 2nd of Jan, ...
    end_day = int
    
    Output:
    trip_data = list, [pickup_time / 10, pickup_place, dropoff_time / 10, dropoff_place]
    
    Note, time expressed in deciseconds. Places are nodes in the osmxn graph. Times are counted relative 
    to the start of start_day
    """
    
    trip_data = []   #will store (pickup place, pickup time)

    #Needs to convert node id's to the osmnx id's
    path = '/home/kokeeffe/research/robocab/'
    keys = np.loadtxt(path+'data/giovanni_to_os_keys.txt')
    values = np.loadtxt(path+'data/giovanni_to_os_values.txt')
    d = {int(key):int(value) for key,value in zip(keys,values)}

    #Read in data
    with open(path+'data/taxi_id.csv') as g:
        reader = csv.reader(g)
        ctr = 0
        for row in reader:
            time_temp = int(row[1])
            datetime_temp = datetime.datetime.fromtimestamp(time_temp)
            day = datetime_temp.day
            if day < start_day or day == 31:  #there are some trips on 31st of December.
                pass
            elif day > end_day:
                break
            else:
                if ctr == 0:
                    t0 = copy.copy(int(row[1]))

                    pickup_time, pickup_place = 0, int(row[3])
                    dropoff_time, dropoff_place = int(row[2]), int(row[4])

                    pickup_place, dropoff_place = d[pickup_place], d[dropoff_place]  #convert to osmnx id's
                    dropoff_time = dropoff_time - t0
                    ctr += 1
                else:
                    pickup_time, pickup_place = int(row[1]), int(row[3])
                    dropoff_time, dropoff_place = int(row[2]), int(row[4])

                    pickup_time, dropoff_time = pickup_time - t0, dropoff_time - t0
                    pickup_place, dropoff_place = d[pickup_place], d[dropoff_place]

                #Append to data
                if pickup_place in G.nodes() and dropoff_place in G.nodes(): 
                    trip_data.append([pickup_time / 10, pickup_place, dropoff_time / 10, dropoff_place])
    trip_data = np.array(trip_data)  #This form is neater for later
    return trip_data





def rhs(x, p, A):
    """ 
    
    Input:
    x = np.array, waiting times
    p = np.array, probabilities
    A = np.array, adj matrix
    
    Output:
    np.array, p + (1-p)*(1 + min_(neighbours i) x_j) 
    """
    
    b = A*x
    temp = np.array([min(i for i in row if i > 0) for row in b])
    return p + (1-p)*(np.ones_like(x) + temp)



def find_optimal_waiting_times(x,p,A, tolerance = 0.01, verbose=False):
    """ Looks for a fixed point to
    
        x = rhs(x),  where rhs is the function defined about / elsewhere
    
        Input:
        x = np.array, waiting times initial guess
        p = np.array, probabilities
        A = np.array, adj matrix
        tolerance = float, when to terminate the recursion.
    
        Output:
        x = np.array, waiting times true / converged
    
    """
    
    err = 1
    xold = x
    while err > tolerance:
        xnew = rhs(xold,p,A)
        err = (xnew - xold).max()
        xold = xnew
        if verbose == True:
            print 'max erro = ' + str(err)
    return xnew



def extract_policy(x,G,nodes):
    """ Given waiting times, x, extract the policy
    
        Input:
        x = np.array, optimal waiting times
        G = osmnx graph, steet network
        nodes = list, [node for node in G.nodes()]
        
        Output:
        policy = dict, where policy[i] = j means when at node i go to node j (i,j, osmnx labels of nodes)
    
    """
    
    policy = {}  # when at node i go to node j, where policy[i] = j   (zero-based)
    for node in nodes:
        neighbours = G[node].keys()
        neighbours_indices = [nodes.index(i) for i in neighbours]
        xs = x[neighbours_indices]
        index = np.argmin(xs)
        node_index = neighbours_indices[index]
        destination = nodes[node_index]
        policy[node] = destination
    return policy



def find_optimal_policy(p,G):
    """
    Find the optimal policy (Sam's result) for a streetnetwork
    
    Input:
    p = np.array, node proabilities (prob of trip generated at node i)
    G = osmnx graph, street network
    
    Output:
    policy = dict, {node_current:node_next}, i.e. when at node A go to node B
    
    """
    
    G = nx.DiGraph(G)
    A = nx.adjacency_matrix(G)
    A = np.array(A.todense())
    nodes = [node for node in G.nodes()]

    x0 = np.random.rand(len(p))
    x = find_optimal_waiting_times(x0,p,A)

    policy = extract_policy(x,G,nodes)
    
    return policy




def find_trip_probs(G,trip_data):
    """ Finds the find_trip_probs, the
        probability of a trip 'being born'
        at each node
        
        Input:
        G = osmnx graph, street network
        trip_data = [t_pickup, x_pickup, t_dropoff, x_dropoff ]
                    where t_pickup are in deci-seconds, and x_i are osmnx ids
    
    """
    
    node_list = [node for node in G.nodes()]

    #Initialize
    trip_probs = {}           #{place, count}
    for node in G.nodes:
        trip_probs[node] = 0

    for pickup_time,pickup_place,dropoff_time,dropoff_place in trip_data:
        trip_probs[pickup_place] += 1

    # This is a day's worth of trips, so divide by 86400 to get the rate
    for key in trip_probs:
        trip_probs[key] /= 8640.0  #turn into an instantaneous rate
        
    return trip_probs



def find_greedy_policy(trip_probs,G):
    """ Finds the greedy policy, where you go
        to node with highest p
        
        Input:
        G = osmnx graph, street network
        trip_probs = dict, counts[node] = p_i, where p_i = trip generation prob, node = osmnx id
        
        Output:
        greedy_policy = dict, greedy_policy[node_curretn] = node_next, node = osmnx id
        
    """
    
    
    greedy_policy = {}
    for node in trip_probs:
        neighbours = G[node].keys()
        ps = [trip_probs[i] for i in neighbours]
        max_index = np.argmax(ps)
        destination = neighbours[max_index]
        greedy_policy[node] = destination
    return greedy_policy




def find_random_policy(G):
    """ Finds a random policy, 
        
        Input:
        G = osmnx graph, street network
        
        Output:
        random_policy = dict, greedy_policy[node_curretn] = node_next, node = osmnx id
        
    """
    
    
    random_policy = {}
    for node in trip_probs:
        neighbours = G[node].keys()
        random_index = np.random.choice(range(len(neighbours)))
        destination = neighbours[random_index]
        random_policy[node] = destination
    return random_policy



def extract_Q_policy(cab,G):
    """ Given a Q-cab, extracts the
        optimal policy.
        
        Input:
        Qcab = instance of Qcab class
        G = osmnx graph, street network
        
        Output:
        policy = dict, policy[node_current] = node_next, where nodes are labeld by osmnx id
    """
    
    policy = {}
    for node in cab.Q:
        Qmax_index = np.argmax(cab.Q[node])
        neighbours = G[node].keys()
        destination = neighbours[Qmax_index]
        policy[node] = destination
    return policy



def get_gps_coords_of_node(G,node):
    """
    Input:
    G = osmnx graph, street network
    node = int, osmnx id
    """
    
    lat = G.nodes[node]['x']
    lon = G.nodes[node]['y']
    return (lat,lon)



def slide(env,G):
    """
    Makes of a plot of the graph G, with taxi
    show as a green dot, and trips shown as
    red dots
    """
    
    fig, ax = ox.plot_graph(G, show=False, close=False)
    
    #Plot taxi
    fig, ax = ox.plot_graph(G, show=False, close=False)
    (lat,lon) = get_gps_coords_of_node(G,env.state)
    ax.scatter(lat,lon, c='green', s=100)
    
    #Plot trips
    for trip in env.available_trips:
        (lat,lon) = get_gps_coords_of_node(G,trip[1])
        ax.scatter(lat,lon, c='red')

        
class Modelcab:
    """ 
    Instance of a cab which follows our "model policy",
    which is Sam's analytic result
    """
    
    
    def __init__(self,policy):
        self.policy = policy
        
    def act(self,state):
        action = self.policy[state]
        return action  