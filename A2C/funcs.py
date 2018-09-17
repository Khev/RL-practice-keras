""" Functions I developed for the taxi graph problems """


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
import copy


def make_G(n):
    """
    Makes an nxn grid graph, where the nodes
    have a random 'prob' state variable, which
    characterizes the odds of a ride being
    generated at that node
    """
    G = nx.grid_graph([n,n])
    G = nx.convert_node_labels_to_integers(G)
    np.random.seed(1)
    ps = np.random.rand(n**2)
    for i,p in enumerate(ps):
        G.nodes[i]['prob'] = p
    return G



def load_manhattan_graph():
    
    """  
    Loads in manhattan graph. The nodes 
    and edges for this graph are specified 
    in the adjacency.csv document. 
    
    The trip generation probability for each
    node is in the probs.csv files
    """

    G = nx.Graph()
    path = '/home/kokeeffe/research/robocab/'
    with open(path + 'data/adjacency.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            node1,node2,length = int(row[1]), int(row[2]), row[3]
            if node1 not in G.nodes():
                G.add_node(node1)
            if node2 not in G.nodes():
                G.add_node(node2)
            G.add_edge(node1,node2)
                    
    with open(path + 'data/sample_probs.csv') as f:
        reader = csv.reader(f)
        ctr = 1
        nodes = list(G.nodes())
        for row in reader:
            G.node[ctr]['prob'] = float(row[0])
            ctr += 1
    return G



def play_game(num_steps):
    """  Play a game of robo cap! """
    
    #Initialize
    num_nodes = 3
    G = make_G(num_nodes)
    taxi = random_taxi(0,0)
    
    #Play game
    for i in range(num_steps):
        taxi.step(G)
    idle_time = taxi.waiting_time / (1.0*taxi.active_time)
    return idle_time



class random_cab:

    def __init__(self,name,start):
        self.name = name
        self.position = start
        self.idle_time = 0
        self.active_time = 0
        
    def move(self,G):
        neighbours = G[self.position].keys()
        self.position = int(np.random.choice(neighbours))
        
    def step(self,G):
        """ Take a step in the game """
        
        #Do action
        self.move(G)
        
        #Generate reward
        temp = np.random.rand()
        p_current = G.nodes[self.position]['prob']
        if temp <= p_current: # Someone wants a ride
            
            #Find the destination
            #print 'Some wants a ride!'
            nodes = list(G.nodes())
            nodes.remove(self.position)
            destination = np.random.choice(nodes)
            
            #Move to desination, and find how long it took
            len_path = len(nx.shortest_path(G,source=self.position,target=destination))
            self.active_time += len_path
            self.position = destination
    
            
        else: #No ride -- keep looking
            #print 'No ride! Keep looking'
            self.move(G)
            self.idle_time += 1
            self.active_time += 1
            
            
    def move_till_find_trip(self,G):
        empty = True
        while empty:
            temp = np.random.rand()
            p_current = G.nodes[self.position]['prob']
            if temp <= p_current:  #found ride
                empty = False
            else: #keep looking
                self.move(G)
                self.idle_time += 1
                self.active_time += 1
            
            
            
class optimal_cab():

    def __init__(self,name,start,smart_moves):
        self.name = name
        self.position = start
        self.idle_time = 0
        self.active_time = 0
        smart_moves_temp = smart_moves   #defined for NYC graph only
        self.smart_moves = [int(i) for i in smart_moves_temp]
        
        # smart_moves[i] = the node you should go to when you're at node i (i.e. the smart move)
        # As sam computed with his model

    
    def move(self,G):
        self.position = self.smart_moves[self.position-1]  #smart_moves is zero based
        
    def step(self,G):
        """ Take a step in the game """
        
        #Do action
        self.move(G)
        
        
        #Generate reward
        temp = np.random.rand()
        p_current = G.nodes[self.position]['prob']
        if temp <= p_current: # Someone wants a ride
            
            #Find the destination
            #print 'Some wants a ride!'
            nodes = list(G.nodes())
            nodes.remove(self.position)
            destination = np.random.choice(nodes)
            
            #Move to destination, and find how long it took
            len_path = len(nx.shortest_path(G,source=self.position,target=destination))
            self.active_time += len_path
            self.position = destination
            
        else: #No ride -- keep looking
            #print 'No ride! Keep looking'
            self.idle_time += 1
            self.active_time += 1
            
            
    def move_till_find_trip(self,G):
        empty = True
        while empty:
            temp = np.random.rand()
            p_current = G.nodes[self.position]['prob']
            if temp <= p_current:  #found ride
                empty = False
            else: #keep looking
                self.move(G)
                self.idle_time += 1
                self.active_time += 1
                
                
                
                
class Qcab_epsilon_greedy:
    def __init__(self,name,start):
        self.name = name
        self.position = start
        self.idle_time = 0
        self.active_time = 0
        self.reward_list = []
        self.trajectory = [start]
        
        
        
    def step(self,G,epsilon,round_number):
        
        #Choose action
        old_position = copy.copy(self.position)
        Qs = find_Qs(G,self.position)
        neighbours = G[self.position].keys()
        temp = np.random.rand()
        if temp < epsilon:
            action_index = np.random.choice(range(len(Qs)))
        else:
            action_index = np.argmax(Qs)
        action = neighbours[action_index]
        self.position = action
        self.trajectory.append(self.position)
        #print old_position, self.position

        #Generate reward
        p_current = G.nodes[self.position]['prob']
        temp = np.random.rand()
        if temp <= p_current: # There is a ride
            reward = 1
            nodes = list(G.nodes())
            nodes.remove(self.position)
            destination = np.random.choice(nodes)
            len_path = len(nx.shortest_path(G,source=self.position,target=destination))
            self.active_time += len_path
            self.position = destination
        else:
            reward = 0
            self.idle_time += 1
            self.active_time += 1

        #Update Q table
        self.reward_list.append(reward)
        Q_old = G[old_position][action]['Q']
        Q_new = Q_old + (1.0/(1.0+round_number))*(reward - Q_old)
        G[old_position][action]['Q'] = Q_new
        
        
    def step_Qgreedy(self,G,Qmoves):
        """ Take a Q-greedy step in the game
            without updating the Q table.
        """
        
        #Do action
        self.position = Qmoves[self.position-1]  #zero to one based
        
        #Generate reward
        temp = np.random.rand()
        p_current = G.nodes[self.position]['prob']
        if temp <= p_current: # Someone wants a ride
            
            #Find the destination
            #print 'Some wants a ride!'
            nodes = list(G.nodes())
            nodes.remove(self.position)
            destination = np.random.choice(nodes)
            
            #Move to destination, and find how long it took
            len_path = len(nx.shortest_path(G,source=self.position,target=destination))
            self.active_time += len_path
            self.position = destination
            
        else: #No ride -- keep looking
            #print 'No ride! Keep looking'
            self.idle_time += 1
            self.active_time += 1
        
        
    def episode(self,num_steps,G,epsilon,start):
        for i in range(num_steps):
            self.step(G,epsilon,i)
        rel_idle_time = copy.copy((1.*self.idle_time) / self.active_time)
        self.idle_time = 0
        self.active_time = 0
        self.position = start
        return rel_idle_time
    
    
    def episode_Qgreedy(self,num_steps, G):
        Qmoves = extract_smart_moves(G)
        Qmoves = [int(i) for i in Qmoves]
        for i in range(num_steps):
            self.step_Qgreedy(G,Qmoves)
        rel_idle_time = copy.copy((1.*self.idle_time) / self.active_time)
        return rel_idle_time
            
              
            
def play_game_randomcab(num_steps,G):
    """  Play a game of robo cap! """
    
    name = 0
    nodes = list((G.nodes()))
    initial_position = np.random.choice(nodes)
    taxi = random_cab(name,initial_position)

    #Play game
    for i in range(num_steps):
        taxi.step(G)
    rel_idle_time = taxi.idle_time / (1.0*taxi.active_time)
    return rel_idle_time



def play_game_optimalcab(num_steps,G, smart_moves):
    """  Play a game with optimal cab"""
    
    name = 0
    nodes = list((G.nodes()))
    initial_position = np.random.choice(nodes)
    taxi = optimal_cab(name,initial_position,smart_moves)

    #Play game
    for i in range(num_steps):
        taxi.step(G)
    rel_idle_time = taxi.idle_time / (1.0*taxi.active_time)
    return rel_idle_time




def play_game_compare(num_steps,G):
    """  Compare robocab to random cab"""
    
    
    #np.random.seed(0)
    name = 'robocab'
    nodes = list((G.nodes()))
    initial_position = np.random.choice(nodes)
    robocab = optimal_cab(name,initial_position, optimal_moves)
    
    name = 'randomcap'
    initial_position = np.random.choice(nodes)
    randomcab = random_cab(name,initial_position)
    

    #Play game
    for i in range(num_steps):
        randomcab.step(G)
        robocab.step(G)
    random_idle = randomcab.idle_time / (1.0*randomcab.active_time)
    robocab_idle = robocab.idle_time / (1.0*robocab.active_time)
    return random_idle, robocab_idle



def find_score(G,optimal_moves):
    """ Compares the strategy vectors
        of the Q-learner, Q_moves, to 
        the optimal ones, as computed by Sam.
        
        The score I'm using is the relative 
        number of agreements
    """
    
    Qmoves = extract_smart_moves(G)
    hits = [1 if i==j else 0 for i,j in zip(Qmoves,optimal_moves)]
    score = (1.0*sum(hits)) / len(hits)
    return score



def find_Qs(G,position):
    Qs = np.array([i['Q'] for i in G[position].values()])
    return Qs



def count_num_zero_Qvals(G):
    num_zeros = 0
    for node in G.nodes():
        Qs = find_Qs(G,node)
        temp = sum([1 if i == 0 else 0 for i in Qs])
        num_zeros += temp
    return num_zeros



def add_Qs_to_graph(G):
    """
    For the Q-learning problem:
    I'm storing the Q value in the graph G,
    Here I'm just filling them in
    """
    for node in G.nodes():
        neighbours = G[node].keys()
        for neighbour in neighbours:
            G[node][neighbour]['Q'] = 0
    return G


def find_greedy_move(G):
    greedy_moves = np.zeros(len(G))
    for i,node in enumerate(G.nodes()):
        greedy_move = G[node].keys()[np.argmax([G.nodes[j]['prob'] for j in G[node].keys()])]
        greedy_moves[i] = greedy_move
    return greedy_moves


def extract_smart_moves(G):
    """ Find the moves the current 
        smart path: i.e. the best action 
        for every state
    """
    
    path = np.zeros(len(G))
    for i,node in enumerate(G.nodes()):
        Qs = find_Qs(G,node)
        neighbours = G[node].keys()
        path[i] = neighbours[np.argmax(Qs)] #nodes are 1 based
    return path


class Qcab:
    def __init__(self,name,start):
        self.name = name
        self.position = start
        self.idle_time = 0
        self.active_time = 0
        self.reward_list = []
        
    def step(self,G,lr,y,round_number):
        old_position = copy.copy(self.position)
        Qs = find_Qs(G,self.position)
        neighbours = G[self.position].keys()
        noise_strength = 1.0/(round_number+1)  #zero based. I want less randomness as the game goes on
        action_index = np.argmax(Qs + noise_strength*np.random.random(len(Qs)))
        action = neighbours[action_index]
        self.position = action
        #print old_position, self.position

        #See if there is a ride there
        p_current = G.nodes[self.position]['prob']
        temp = np.random.rand()
        if temp <= p_current: # There is a ride
            reward = 1
            nodes = list(G.nodes())
            nodes.remove(self.position)
            destination = np.random.choice(nodes)
            len_path = len(nx.shortest_path(G,source=self.position,target=destination))
            self.active_time += len_path
            self.position = destination
        else:
            reward = 0
            self.idle_time += 1
            self.active_time += 1

        #Update Q table
        self.reward_list.append(reward)
        Qs_new = find_Qs(G,self.position)
        update = lr*(reward + y*np.max(np.max(Qs_new) - Qs[action_index]))
        #print update
        G[old_position][action]['Q'] += update
        
        
        
def draw_graph_with_smart_moves(G):
    """
    Draws the graph G, with the 'smart-moves'
    i.e. which edge the Q-learner takes
    drawn in red
    
    The size of the nodes are proportional to their
    trip probability
    
    """
    
    #Extract the smart moves
    smart_moves = extract_smart_moves(G)
    smart_moves_temp = [(i,int(smart_moves[i])) for i in range(len(smart_moves))]
    smart_moves_temp = [(a,b) if a<b else (b,a) for (a,b) in smart_moves_temp]  #reorder edges
    
    #Make colors
    labels = [i for i in range(1,len(G.nodes())+1)]
    edges = G.edges()
    edge_colors = ['r' if edge in smart_moves_temp else 'k' for edge in edges]
    node_colors = [200*G.nodes[i]['prob'] for i in range(len(G.nodes))]
    pos = nx.spring_layout(G) 
    nx.draw(G,pos,edge_color=edge_colors,node_size=node_colors)
    
    
    
def draw_graph_with_optimal_moves(G,optimal_moves):
    """
    Draws the graph G, with the optimal edges drawn in
    The size of the nodes are proportional to their
    trip probability
    """
    
    #Extract the smart moves
    smart_moves = optimal_moves
    smart_moves_temp = [(i,int(smart_moves[i])) for i in range(len(smart_moves))]
    smart_moves_temp = [(a,b) if a<b else (b,a) for (a,b) in smart_moves_temp]  #reorder edges
    
    #Make colors
    labels = [i for i in range(1,len(G.nodes())+1)]
    edges = G.edges()
    edge_colors = ['r' if edge in smart_moves_temp else 'k' for edge in edges]
    node_colors = [200*G.nodes[i]['prob'] for i in range(len(G.nodes))]
    pos = nx.spring_layout(G) 
    nx.draw(G,pos,edge_color=edge_colors,node_size=node_colors)
    
    
    
  
        
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




def find_greedy_policy(G):
    policy = {}
    for node in G.nodes:
        neighbours = G[node].keys()
        probs = [G.nodes[neighbour]['prob'] for neighbour in neighbours]
        max_index = np.argmax(probs)
        policy[node] = neighbours[max_index]
    return policy



def make_greedy_cabs(num_cabs,G):
    greedy_policy = find_greedy_policy(G)
    return [Cab(i, np.random.choice(G.nodes()),greedy_policy) for i in range(num_cabs)]




class Cab:
    
    def __init__(self,name,position,policy):
        self.name = name
        self.position = position
        self.policy = policy
        self.state = 'searching'
        self.route = []   #route to take to serve passenger
        self.active_time = 0
        self.idle_time = 0
        
        
    def step(self):
        if self.state == 'serving':
            if len(self.route) == 1:
                self.position = self.route[0]
                self.route = []
                self.active_time += 1
                self.state = 'searching'
            else:
                self.position = self.route[0]
                self.route = self.route[1:]
                self.active_time += 1
        else:
            self.position = self.policy[self.position]
            self.active_time += 1
            self.idle_time += 1




class Evn_gridgraph:
    
    def __init__(self,G,cabs):
        self.cabs = cabs
        self.G = G        
    
    def step(self):

        #Move cabs
        density = {}  # position:[cab1, cab2, cab3, ...] I need this alter
        for cab in self.cabs:
            cab.step()
            
            if cab.state == 'serving':
                pass
            else:
                if cab.position not in density:
                    density[cab.position] = [cab]
                else:
                    density[cab.position].append(cab)
                    
        
        #Generate rewards
        for node in density:
            temp = np.random.rand()
            if temp <= self.G.nodes[node]['prob']:  #a trip is generated
                                
                #Pick which cab gets trip
                num_cabs_at_node = len(density[node])
                cab_index = np.random.choice(range(num_cabs_at_node))  
                density[node][cab_index].state = 'serving'
                
                #Find destination
                nodes = list(self.G.nodes())
                nodes.remove(density[node][cab_index].position)
                destination = np.random.choice(nodes)
                
                #Find route to that destination
                route = nx.shortest_path(self.G,source=density[node][cab_index].position,target=destination)
                route = route[1:]  #remove starting node
                density[node][cab_index].route = route
                
                #Change state
                density[node][cab_index].state = 'serving'
                
                
    
    def step_independent(self):
        """ Here cabs don't interact with each other """
                
        for cab in self.cabs:  
            
            #Action
            cab.step() 
            
            #Reward
            if cab.state == 'serving':
                pass
            else:
                temp = np.random.rand()  #generate reward
                node = copy.deepcopy(cab.position)
                if temp <= self.G.nodes[node]['prob']:  #a trip is generated
                    
                    #Find destination
                    nodes = list(self.G.nodes())
                    nodes.remove(node)
                    destination = np.random.choice(nodes)

                    #Find route to that destination
                    route = nx.shortest_path(self.G,source=node,target=destination)
                    route = route[1:]  #remove starting node
                    cab.route = route
                    
                    #Change state
                    cab.state = 'serving'
                    
                    
                    
    def episode(self,num_steps):
        for j in range(num_steps):
            self.step()
            
            
    def episode_independent(self,num_steps):
        for j in range(num_steps):
            self.step_independent()
    
        
                
def make_random_policy(G):
    policy = {}
    for node in G.nodes():
        neighbours = G[node].keys()
        destination = np.random.choice(neighbours)
        policy[node] = destination
    return policy


def make_cabs(num_cabs,G):
    cabs = []
    for i in range(num_cabs):
        policy = make_random_policy(G)
        position = np.random.choice(G.nodes())
        cabs.append(Cab(i,position,policy))
    return cabs


def find_taus(cabs):
    return [(1.0*cab.idle_time) / cab.active_time for cab in cabs]



def breed(policy1, policy2):
    new_policy = copy.deepcopy(policy1)
    for node in policy1:
        rand = np.random.rand()
        if rand > 0.5:
            new_policy[node] = policy1[node]
        else:
            new_policy[node] = policy2[node]
    return new_policy



def mutate(policy,G,p=0.05):
    new_policy = copy.deepcopy(policy)
    for node in policy:
        rand = np.random.rand()
        if rand < p:
            neighbours = G[node].keys()
            new_policy[node] = np.random.choice(neighbours)
        else:
            new_policy[node] = policy[node]
    return new_policy


def selection(cabs, cutoff=5):
    taus = find_taus(cabs)
    elite_indices = list(np.argsort(taus))
    elite_policies = [cabs[i].policy for i in elite_indices[:cutoff]]
    return elite_policies



def evolve(cabs,G,cutoff = 5, mutation_rate = 0.05):
    elite_policies = selection(cabs, cutoff=cutoff)
    for cab in cabs:
        new_policy = breed(np.random.choice(elite_policies), np.random.choice(elite_policies))
        new_policy = mutate(new_policy,G, p = mutation_rate)
        cab.policy = new_policy
        cab.active_time = 0
        cab.idle_time = 0
    return None




def slide(cabs,G,pos):
    """
    Plots graph, with empty cabs shown as 'E'
    and full cabs as "F".
    
    
    Input:
    cabs = list, [cab1, cab2, ...] where cab is an instance of my Cab class
    G = nx network, street network
    
    pos = nx.spring_layout(G)  # for plotting,easier to define this outside the functions, since
                                 its a non-deterministic algorithm -- so slides in neighboring 
                                 timesteps will not look the same (nodes layed out differently)

    """
        
    plt.figure()
    

    #Make colors
    labels = [i for i in range(1,len(G.nodes())+1)]
    nodes = G.nodes()
    #node_colors = ['blue' if cab.state == 'serving' else 'red' for cab in cabs]
    node_colors = ['grey' for cab in cabs]
    alphas = [0.1 for i in G.nodes()]
    node_labels = {cab.position:str(cab.name)+'F' if cab.state == 'serving' else str(cab.name)+'E' \
                   for cab in cabs}
    sizes = {10 for node in G.nodes()}
    #pos = nx.spring_layout(G) 
    nx.draw(G,pos,node_color=node_colors,alpha=0.9,labels=node_labels,size=sizes,font_size=20,
           node_size=500)
    
    
def find_tau_given_policy(cab,G,num_steps):
    temp_cab = copy.deepcopy(cab)
    temp_env = Evn_gridgraph(G,[temp_cab])
    temp_env.episode_independent(num_steps)
    return temp_cab.idle_time / (1.0*temp_cab.active_time)
