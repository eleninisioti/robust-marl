""" This script contains various helper functions.
"""
from node import *
import random
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import LinAlgWarning
import warnings
warnings.filterwarnings("error", category=LinAlgWarning)

import gurobipy

def solve_LP(num_a, num_o, Qtable):

  c = np.zeros((num_a + 1, 1))
  c[0] = -1
  A_ub = np.ones((num_o, num_a + 1))

  A_ub[:, 1:] = -np.reshape(Qtable, A_ub[:, 1:].shape)
  b_ub = np.zeros((num_o, 1))
  A_eq = np.ones((1, num_o + 1))
  A_eq[0, 0] = 0
  b_eq = [1]
  bounds = ((None, None),) + ((0, 1),) * num_a

  counter = 0
  while counter < 100:

    try:
      counter += 1
      res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds)

    except LinAlgWarning:
      print("Error: result is inaccurate due to illconditioning.")
    else:
      break

    if counter == 99:
      print("Error: Optimisation failed.")

      # solve LP even if it is illconditioned
      warnings.filterwarnings("default", category=LinAlgWarning)

      res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds)
      warnings.filterwarnings("error", category=LinAlgWarning)

  return res


def find_adversarial_policy(agents, attack_size):
  """ Finds the optimal adversarial policy for one-step attacks of size
  attack_size.

  Args:
    agents (list of `obj`:Agent): agents, each one has its own policy
    attack_size (K): number of attackers

  Returns:
    adversarial policy
  """

  # get policies of all agents
  policies = []
  for idx, agent in enumerate(agents):
    policies.append(agent.policies)
  policies = [item for sublist in policies for item in sublist]


  # get all possible partitions
  nodes = []
  for agent in agents:
    nodes.extend(agent.control_nodes)

  indexes = list(range(len(nodes)))
  attackers_subsets = list(itertools.combinations(indexes, attack_size))


  # get current state
  state_space = agents[0].state_space # assumes all agents have the same space
  states = []
  for state_col in state_space:
    states.append(list(range(state_col)))

  # initialize adversarial policy for classical actions
  actions_space = []
  for adv in range(attack_size):
    actions_space.extend([2])

  sigma_actions =  np.zeros(tuple(state_space + actions_space))

  # initialize adversarial policy for choosing nodes
  partitions_space = []
  for adv in range(attack_size):
    partitions_space.append(1)
  sigma_partitions =  np.zeros(tuple(state_space + partitions_space))

  for state in itertools.product(*states):

    # get Qvalue for current state
    current_entry = [slice(None)] * len(agent.state_space)
    for idx, el in enumerate(state):
      current_entry[idx] = el
    Qcurrent = agent.Qtable[tuple(current_entry)]

    min_Qvalue = np.abs(np.max(Qcurrent))*999
    worst_partition = []
    worst_adv_action = []
    for attackers in attackers_subsets:
      attackers = list(attackers)
      defenders = [idx for idx, node in enumerate(nodes) if idx not in
                                                           attackers]

      # get all possible action combinations (for marginalisation)
      policies_product= []
      indices  = []

      for policy_idx, policy in enumerate(policies):

        if policy_idx in defenders:
          policies_product.append(np.ndarray.tolist(np.ndarray.flatten(policy)))
          indices_flat = list(range(len(np.ndarray.flatten(policy))))

          indices.append(list(np.unravel_index(indices_flat,
                                               policy.shape)))

      Qdefend = []
      counter = 0
      indices_product = list(itertools.product(*indices))
      for policy_list in itertools.product(*policies_product):
        #policy_term = np.prod(policy_list)

        counter=0
        indices_map = [[0,0], [0,1], [1,0],[1,1]]
        for policy_term in policy_list:
          current_indices = indices_map[counter]

          current_entry = [slice(None)] * len(agent.action_space)
          for idx, defender_idx in enumerate(defenders):
            current_entry[defender_idx*2] = current_indices[idx]
            current_entry[defender_idx*2+1] = current_indices[idx+1]
          Qdefend.append(Qcurrent[tuple(current_entry)]*policy_term)
          counter+=1

      # marginalize over all compute Qtables
      total_Qdefend = np.zeros(Qcurrent[tuple(
          current_entry)].shape)
      for table in Qdefend:
        total_Qdefend = np.add(total_Qdefend, table)

      adv_action = np.argmin(total_Qdefend)
      adv_action = list(np.unravel_index(adv_action,
                            total_Qdefend.shape))

      # keep worst attack
      if np.min(total_Qdefend) <= min_Qvalue:
        worst_partition = attackers
        worst_adv_action = adv_action
        min_Qvalue = np.min(total_Qdefend)

    # update adversarial policy
    current_entry = [slice(None)]*len(state_space)
    for idx,el in enumerate(state):
      current_entry[idx] = el

    if len(worst_adv_action) < 1:
      print("oops")

    sigma_actions[tuple(current_entry)] =  worst_adv_action
    sigma_partitions[tuple(current_entry)] = worst_partition

  sigma_actions = np.squeeze(sigma_actions) # in case we assumed too many
  # neighbors
  print(sigma_partitions, sigma_actions)

  return [sigma_partitions, sigma_actions]

def env_interact(agents, delta, chigh, clow, utility, K, exploration,
                 attack_type, current_state=[], adversarial_policy=[]):
  """ Performs an interaction between the agents and the environment.
  
  Args: 
    agents (list of `obj`:Agent): agents
    current_delta (float): between 0 and 1, probability of attack
    K (int): number of adversaries
    exploration (bool): indicates whether exploration should take place

  Returns:
    a list of actions, a list of rewards and a list of new states

  """
  # decide whether an attack takes place during testing
  x = random.uniform(0, 1)
  if x <= delta:
    attack_size = K
  else:
    attack_size = 0

  attack_actions = {}

  if attack_size > 0:
    #print(adversarial_policy)

    attack_actions = perform_attack(adversarial_policy=adversarial_policy,
                                    attack_size=attack_size,
                                    current_state=current_state,
                                    attack_type=attack_type,
                                    agents=agents)

  # find actions performed by agents
  actions = []
  for idx, agent in enumerate(agents):
    agent_actions = agent.execute_policy(attack_actions, exploration)

    # if len(agent_actions) < 4:
    #   print("oops")
    # if isinstance(agent, MinimaxQAgent):
    #   agent_actions = agent_actions[idx*2:(idx*2+2)]

    actions.extend(agent_actions)


  nodes = []
  recipients = []
  served = []
  total_nodes = []

  for agent in agents:
    agent_nodes = agent.control_nodes
    nodes.extend(agent_nodes)
    total_nodes.extend(agent.nodes)

    # get all actions performed by nodes
    for node in agent_nodes:
      node_idx = node.idx
      if len(actions) <4:
        print("oops")
      served.append(actions[(node_idx - 1) * 2])  # assumes that the order
      # of the nodes in actions is the same with the order in
      # nodes, and idxs are in ascending order
      recipient = actions[(node_idx - 1) * 2 + 1]
      recipients.append(recipient)

  # find arrivals and departures for each node
  new_states = []
  rewards = []
  arrivals = []
  departures = []
  for node in nodes:
    node_idx = node.idx
    recipient = recipients[node_idx - 1]

    # ignore transmission if no packets
    if node.load <=0:
      recipient = 0

    # find whether the node has transmitted a packet
    if recipient:
      departures.append(1)
    else:
      departures.append(0)

    # find how many agents sent a packet to this node
    arrivals.append(sum([1 for idx, recipient in enumerate(recipients) if
                         ((recipient == node_idx) and (
                               total_nodes[idx].load > 0))]))

  # ----- experience transitions -----
  stop_episode = False
  for node in nodes:
    arr = arrivals[node.idx - 1]
    dep = departures[node.idx - 1]
    costs = node.costs

    recipient = recipients[node.idx - 1]


    underflow, overflow, served_new = node.transition(arrivals=arr,
                                              departures=dep,
                                              served=served[node.idx - 1])

    if overflow:
      reward = - chigh
      stop_episode = True

    elif underflow:
      reward = - clow
    else:
      reward = utility

    # add costs of transmission and execution
    if recipient:
      transmission_cost = costs[recipient]
      reward -= transmission_cost

    if served_new:
      reward -= node.serve_cost

    rewards.append(reward)
    new_states.append(node.load)


  return actions, rewards, new_states, stop_episode



def create_star_topology(nodes, capacity, cost):
  """ Creates a network of nodes with a star topology.

  All nodes have the same capacity and cost.

  Args:
    nodes (list of :obj:`Node`): all nodes of the network
    capacity (int): capacity of nodes
    cost (float): cost of serving a packet

  TODO: needs debugging
  """
  counter = 0
  neighbors = list(range(len(nodes) - 1))
  capacity = capacity
  costs = {}
  for node in neighbors:
    costs[node] = cost

  nodes.append(Node(capacity, neighbors, idx=counter, costs=costs))
  for idx in range(1, args.N):
    counter += 1
    neighbors = [0]
    capacity = args.capacity
    costs = {0: args.cost}
    nodes.append(Node(capacity, neighbors, idx=counter, costs=costs))

    return nodes


def create_ring_topology(N, capacity, cost):
  """ Creates a network of nodes with a ring topology.


  All nodes have the same capacity and cost.

  Args:
    N (int): number of nodes
    capacity (int): capacity of nodes
    cost (float): cost of serving a packet

  TODO: needs debugging
  """
  nodes = []
  counter = 1
  for idx in range(1, N + 1):
    left = idx - 1
    if left < 1:
      left = N
    right = idx + 1
    if right > N:
      right = 1
    neighbors = [0, left, right]

    # define capacity and costs for this node
    capacity = capacity  # all nodes equal
    costs = {left: cost, right: cost}
    quick_node = False
    nodes.append(Node(capacity=capacity, neighbors=neighbors, idx=counter, \
                      costs=costs,
                      quick_node=quick_node))
    counter += 1

    return nodes


def create_pair(type, capacity):
  """ Creates a toy-network of two nodes.

  Args:
    type (string): choose among predefined types, which determine all
    characteristics, except for capacity

    capacity (int): capacity of nodes

  Returns:
    a list of nodes
  """
  if type == "A":
    costs_1 = {0: 0, 2: 1}
    costs_2 = {0: 0, 1: 1}
    serve_cost_1 = 1
    serve_cost_2 = 1
    spawn_rate_1 = 0.5
    spawn_rate_2 = 0.5
  elif type == "B":
    costs_1 = {0: 0, 2: 5}
    costs_2 = {0: 0, 1: 1}
    serve_cost_1 = 1
    serve_cost_2 = 1
    spawn_rate_1 = 0.7
    spawn_rate_2 = 0

  elif type == "C":
    costs_1 = {0: 0, 2: 3}
    costs_2 = {0: 0, 1: 3}
    serve_cost_1 = 8
    serve_cost_2 = 1  # # was 1
    spawn_rate_1 = 0.7
    spawn_rate_2 = 0.5

  elif type == "D":
    costs_1 = {0: 0, 2: 3}
    costs_2 = {0: 0, 1: 3}
    serve_cost_1 = 5
    serve_cost_2 = 1  # # was 1
    spawn_rate_1 = 0.7
    spawn_rate_2 = 0.5

  elif type == "R":
    costs_1 = {0: 0, 2: 3}
    costs_2 = {0: 0, 1: 3}
    serve_cost_2 = 5
    serve_cost_1 = 1  # # was 1
    spawn_rate_2 = 0.7
    spawn_rate_1 = 0.5

  nodes = []

  neighbors = [0, 2]
  node = Node(capacity=capacity, neighbors=neighbors, idx=1, \
              costs=costs_1, serve_cost=serve_cost_1, spawn_rate=spawn_rate_1)
  nodes.append(node)
  neighbors = [0, 1]
  node = Node(capacity=capacity, neighbors=neighbors, idx=2, \
              costs=costs_2, serve_cost=serve_cost_2, spawn_rate=spawn_rate_2)
  nodes.append(node)

  return nodes

def perform_attack(adversarial_policy, current_state, attack_size,
                   attack_type, agents):
  """ Simulates an attack by adversaries.

  Args:
     adversarial_policy (list of arrays): contains one policy for selecting
     nodes and one policy for selecting actions on behalf of them

     current_state (list of int): the current state of the network

     attack_type: if worst, the adversarial policy is used, otherwise the
     attack is random
  Adversaries choose the agents to attack and perform actions on their behalf.
  """
  attack_actions = {}

  #print(attack_type)
  # ----- random attack -----
  if attack_type == "randoma":

    attackers = random.sample(agents, attack_size)

    nodes = [agent.nodes for agent in attackers]
    nodes_flat = [item for sublist in nodes for item in sublist]
    if len(nodes_flat) > attack_size:
      nodes = random.sample(nodes_flat, attack_size)
    #nodes_flat = [item for sublist in nodes for item in sublist]
    state_adv_nodes = [node.idx -1 for node in nodes]
    state_adv_actions = []
    for item in nodes:
      idx = item.idx -1
      serve = random.randint(0,1)
      send = random.randint(0, len(item.neighbors)-1)
      state_adv_actions.append([serve, send])

  elif attack_type == "randomb":
    adversarial_nodes = adversarial_policy[0]
    state_size = len(current_state)

    current_entry = [slice(None)] * state_size
    for idx, el in enumerate(current_state):
      current_entry[idx] = el

    state_adv_nodes = adversarial_nodes[tuple(current_entry)]
    attackers = random.sample(agents, attack_size)

    nodes = [agent.nodes for agent in attackers]
    nodes_flat = [item for sublist in nodes for item in sublist]
    if len(nodes_flat) > attack_size:
      nodes = random.sample(nodes_flat, attack_size)
    state_adv_actions = []

    for item in nodes:
      idx = item.idx -1
      serve = random.randint(0,1)
      send = random.randint(0, len(item.neighbors)-1)
      state_adv_actions.extend([serve, send])

  elif attack_type == "randomc":
    attackers = random.sample(agents, attack_size)

    nodes = [agent.nodes for agent in attackers]
    nodes_flat = [item for sublist in nodes for item in sublist]
    if len(nodes_flat) > attack_size:
      nodes = random.sample(nodes_flat, attack_size)
    #nodes_flat = [item for sublist in nodes for item in sublist]
    state_adv_nodes = [node.idx -1 for node in nodes]
    adversarial_actions = adversarial_policy[1]
    state_size = len(current_state)

    current_entry = [slice(None)] * state_size
    for idx, el in enumerate(current_state):
      current_entry[idx] = el

    state_adv_actions = adversarial_actions[tuple(current_entry)]

  # ----- adversarial attack -----
  else:

    adversarial_nodes = adversarial_policy[0]
    adversarial_actions = adversarial_policy[1]
    state_size = len(current_state)

    current_entry = [slice(None)] * state_size
    for idx, el in enumerate(current_state):
      current_entry[idx] = el

    state_adv_nodes = adversarial_nodes[tuple(current_entry)]
    state_adv_actions = adversarial_actions[tuple(current_entry)]

  for idx, adv in enumerate(state_adv_nodes):
    adv = int(adv)
    attack_actions[adv] = [int(state_adv_actions[idx]),
                                          int(state_adv_actions[idx+1])]


  return attack_actions


def lp_solve(Q, numactions, opp_numactions, solver="gurobipy"):
  ret = None

  if solver == 'scipy.optimize':
    c = np.append(np.zeros(self.numactions), -1.0)
    A_ub = np.c_[-Q, np.ones(self.opp_numactions)]
    b_ub = np.zeros(self.opp_numactions)
    A_eq = np.array([np.append(np.ones(self.numactions), 0.0)])
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0) for _ in range(self.numactions)] + [(-np.inf, np.inf)]
    res = lib.linprog(
      c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    ret = res.x[:-1]
  elif solver == 'gurobipy':
    Q = Q.T
    m = gurobipy.Model('LP')
    m.setParam('OutputFlag', 0)
    m.setParam('LogFile', '')
    m.setParam('LogToConsole', 0)
    v = m.addVar(name='v')
    pi = {}
    for a in range(numactions):
      pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
    m.update()
    m.setObjective(v, sense=gurobipy.GRB.MAXIMIZE)
    for o in range(opp_numactions):
      m.addConstr(
        gurobipy.quicksum(pi[a] * Q[a, o] for a in range(numactions)) >= v,
        name='c_o{}'.format(o))
    m.addConstr(gurobipy.quicksum(pi[a] for a in range(numactions)) == 1,
                name='c_pi')
    m.optimize()
    ret = np.array([pi[a].X for a in range(numactions)])
  elif solver == 'pulp':
    v = lib.LpVariable('v')
    pi = lib.LpVariable.dicts('pi', list(range(self.numactions)), 0, 1)
    prob = lib.LpProblem('LP', lib.LpMaximize)
    prob += v
    for o in range(self.opp_numactions):
      prob += lib.lpSum(pi[a] * Q[a, o] for a in range(self.numactions)) >= v
    prob += lib.lpSum(pi[a] for a in range(self.numactions)) == 1
    prob.solve(lib.GLPK_CMD(msg=0))
    ret = np.array([lib.value(pi[a]) for a in range(self.numactions)])

  if not (ret >= 0.0).all():
    raise Exception('{} - negative probability error: {}'.format(solver, ret))

  return ret