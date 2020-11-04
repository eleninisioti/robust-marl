""" Contains various misc. functions, used throught a simulation.
"""

# ----- generic imports -----
import random
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import LinAlgWarning
import warnings
warnings.filterwarnings("error", category=LinAlgWarning)

# ----- project-specific imports -----
from node import Node


def solve_LP(num_a, num_o, game_table):
  """ Solves a linear program.

  We assume that the game matrix has opponent actions as a first dimension.

  Args:
    num_a (int): number of player's actions
    num_o (int): number of opponent's actions
    game_table (array of float): the game matrix, a table of dimension (num_o x
    num_a)
    containing the values of different outcomes of the game

  Returns: the solution of the linear program, containing both the value and
  the policy   """


  # defines optimization objective
  c = np.zeros((num_a + 1, 1))
  c[0] = -1

  # inequality constraints
  A_ub = np.ones((num_o, num_a + 1))
  A_ub[:, 1:] = -np.reshape(game_table, A_ub[:, 1:].shape)
  b_ub = np.zeros((num_o, 1))

  # equality contraints
  A_eq = np.ones((1, num_a + 1))
  A_eq[0, 0] = 0
  b_eq = [1]

  bounds = ((None, None),) + ((0, 1),) * num_a # V is unbounded, policy
  # elements in (0,1)

  # solve linear program
  counter = 0
  while counter < 3:
    feasible = True

    try:
      counter += 1
      res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds)
    except ValueError:
      print(ValueError)
      feasible = False

    except:
      print("Error: result is inaccurate due to illconditioning.")
    else:
      break

    if counter == 2:
      if feasible:

        print("Error: Optimisation failed. Reducing accurary")

        # solve LP even if it is illconditioned
        # warnings.filterwarnings("default", category=LinAlgWarning)

        try:

          res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, options={'sym_pos': False,
                                              "cholesky": False,
                                              "lstsq": True})
        except ValueError:
          res = None

      else:
        res = None
      #warnings.filterwarnings("error", category=LinAlgWarning)
  return res


def find_adversarial_policy(agents, attack_size):
  """ Finds the adversarial policy computed based on the optimal policies of
  the MAS.

  Args:
    agents (list of Agent): agents
    attack_size (K): number of attackers

  Returns:
    a list containing the node selection and the action selection (both of
    them consider relative idxs)
  """

  non_admissible = {0: [[1, 1], [1, 0], [0, 1]], 1: [[1, 1]]}
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
  advers_subsets = list(itertools.combinations(indexes, attack_size))

  # get current state
  state_space = agents[0].state_space # assumes all agents have the same space
  states = []
  for state_col in state_space:
    states.append(list(range(state_col)))

  # initialize adversarial policy for action selection
  actions_space = []
  for adv in range(attack_size):
    actions_space.extend([2])
  actions_space = tuple(actions_space)
  sigma_actions = np.zeros(tuple(state_space + actions_space))

  # initialize adversarial policy for node selection
  partitions_space = []
  for adv in range(attack_size):
    partitions_space.append(1)
  partitions_space = tuple(partitions_space)
  sigma_partitions = np.zeros(tuple(state_space + partitions_space))

  for state in itertools.product(*states):

    # get q-value for current state
    current_state = [slice(None)] * len(state_space)
    for idx, el in enumerate(state):
      current_state[idx] = el
    qcurrent = agent.Qtable[tuple(current_state)]

    # initialize variables for searching
    min_qvalue = np.abs(np.max(qcurrent))
    worst_partition = []
    worst_adv_action = []

    # ----- search over all possible partitions -----
    for advers in advers_subsets:
      advers = list(advers)

      # find relative idxs of defenders
      defenders = [idx for idx, node in enumerate(nodes) if idx not in
                                                           advers]

      # ----- get policy values for all possible action combinations -----
      policies_product = []
      indices = []

      for policy_idx, policy in enumerate(policies):

        if policy_idx in defenders:
          policies_product.append(np.ndarray.tolist(np.ndarray.flatten(
            policy[tuple(current_state)])))
          indices_flat = list(range(len(np.ndarray.flatten(policy[tuple(
            current_state)]))))
          ind_temp = list(np.unravel_index(indices_flat, policy[tuple(
            current_state)].shape))
          ind_temp = [np.ndarray.tolist(el) for el in ind_temp]
          indices.append(ind_temp)

      # ----- multiply each policy term with the corresponding Q-value -----
      qdefend = []
      counter_ext = 0
      for policy_list in itertools.product(*policies_product):
        counter=0
        for policy_idx, policy_term in enumerate(policy_list):
          current_indices = [indices[counter][0][counter_ext],
                             indices[counter][1][counter_ext]]
          current_action = [slice(None)] * len(agent.action_space)

          for idx, defender_idx in enumerate(defenders):
            current_action[defender_idx*2] = current_indices[idx]
            current_action[defender_idx*2+1] = current_indices[idx+1]

          # calculate marginal for current policy term
          temp = qcurrent[tuple(current_action)]
          qdefend.append(qcurrent[tuple(current_action)]*policy_term)
          counter += 1

        counter_ext +=1

      # add all qtables (each one corresponds to a different combination of
      # the actions of defenders)
      total_qdefend = np.zeros(qcurrent[tuple(current_action)].shape)
      for table in qdefend:
        total_qdefend = np.add(total_qdefend, table)

      adv_action = np.argmin(total_qdefend)
      adv_action = list(np.unravel_index(adv_action, total_qdefend.shape))
      value = np.min(total_qdefend)

      # ----- for debugging: assume that you play deterministically -----
      # maximize over defenders
      def_actions = np.argmax(qcurrent)
      def_actions = list(np.unravel_index(def_actions, qcurrent.shape))
      act_entry = [slice(None)] * len(def_actions)
      act_entry[defender_idx*2] = def_actions[defender_idx*2]
      act_entry[defender_idx * 2 + 1] = def_actions[defender_idx * 2 +1]
      def_qtable = qcurrent[tuple(act_entry)]

      admissible = False
      while not admissible:
        print("looking for admissible")

        adv_action = np.argmin(def_qtable)
        adv_action = list(np.unravel_index(adv_action, def_qtable.shape))
        value = np.min(def_qtable)

        admissible = True
        if current_state[advers[0]] in non_admissible.keys():
          if adv_action in non_admissible[current_state[advers[0]]]:
            admissible = False
            def_qtable = np.where(def_qtable == value, 999, def_qtable)

        print(adv_action, current_state[advers[0]])

      # check whether the action is admissible
      # keep worst partition
      if value <= min_qvalue:
        worst_partition = advers
        worst_adv_action = adv_action

        # find value of original qtable
        min_qvalue = np.min(total_qdefend)

    # update adversarial policy for current state
    current_state = [slice(None)]*len(state_space)
    for idx,el in enumerate(state):
      current_state[idx] = el

    sigma_actions[tuple(current_state)] = worst_adv_action
    sigma_partitions[tuple(current_state)] = worst_partition

  sigma_actions = np.squeeze(sigma_actions) # in case we assumed too many
  # neighbors
  print("finised")
  return [sigma_partitions, sigma_actions]

def perform_attack(adversarial_policy, current_state, attack_size,
                   attack_type, agents):
  """ Simulates an attack by adversaries.

  Args:
     adversarial_policy (list of arrays): contains one policy for selecting
     nodes and one policy for selecting actions on behalf of them

     current_state (list of int): the current state of the network

     attack_type (str): there's four types of attacks. The default is to follow
     the adversarial policy. In randa, nodes and actions are picked randomly.
      In randb, nodes are picked adversarially and actions
     randomly.
     attack_size (int): number of adversaries
     agents (list of Agent): agents comprising the MAS

  Returns:
    a dictionary where keys are absolute node idxs and actions have relative
    idxs
  """
  attack_actions = {}

  # ----- random attack -----
  if attack_type == "randa":
    # ----- pick both nodes and actions adversarially -----

    # choose agents to attack
    adversaries = random.sample(agents, attack_size)

    # find nodes controlled by attacked agents
    nodes = [agent.nodes for agent in adversaries]
    nodes_flat = [item for sublist in nodes for item in sublist]

    # pick subset of nodes, so that the correct number of nodes is attacked
    if len(nodes_flat) > attack_size:
      nodes = random.sample(nodes_flat, attack_size)

    state_adv_nodes = [node.idx for node in nodes]
    state_adv_actions = []

    for item in nodes:
      serve = random.randint(0,1)
      send = random.randint(0, len(item.neighbors)-1)
      state_adv_actions.append([serve, send])

  elif attack_type == "randb":
    # ----- pick nodes adversarially and actions randomly
    adversarial_nodes = adversarial_policy[0]
    state_size = len(current_state)

    current_entry = [slice(None)] * state_size
    for idx, el in enumerate(current_state):
      current_entry[idx] = el

    state_adv_nodes = adversarial_nodes[tuple(current_entry)]

    state_adv_actions = []
    for item in adversarial_nodes:
      serve = random.randint(0,1)
      send = random.randint(0, len(item.neighbors)-1)
      state_adv_actions.extend([serve, send])

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


def env_interact(agents, prob_attack, payoffs, attack_size,
                 evaluation, attack_type, current_state=[],
                 adv_policy=[]):
  """ Performs an interaction between the agents and the environment.
  
  Args: 
    agents (list of `obj`:Agent): agents
    K (int): number of adversaries
    deployment (bool): indicates whether interaction is during deployment
    prob_attack (float): probability of attack
    payoffs (dict of str: float): payoffs of the game
    attack_type (str): choose between worst, randoma and randomb
    adversarial_policy (list of array): adversarial policy, used when
    attack_type is worst

  Returns:
    a list of actions, a list of rewards and a list of new states
  """
  
  # decide whether an attack takes place
  x = random.uniform(0, 1)
  if x > prob_attack:
    attack_size = 0
  attack_actions = {}

  if attack_size > 0:
    attack_actions = perform_attack(adversarial_policy=adv_policy,
                                    attack_size=attack_size,
                                    current_state=current_state,
                                    attack_type=attack_type,
                                    agents=agents)

  # find actions performed by agents
  actions = []
  nodes = []
  for idx, agent in enumerate(agents):
    agent_actions = agent.execute_policy(attack_actions, evaluation)

    actions.extend(agent_actions)
    nodes.extend(agent.control_nodes)

  # ----- perform one interaction with the environment -----
  recipients = []
  executed = []

  for idx, action in enumerate(actions):
    if idx%2 == 0:
      executed.append(action)
    else:
      recipients.append(action)

  # find arrivals and departures for each node
  new_states = []
  rewards = []
  arrivals = []
  departures = []
  for node in nodes:
    node_idx = node.idx
    recipient = recipients[node_idx - 1]

    # ignore transmissions from this node if it has no tasks
    if node.load <= 0:
      recipient = 0

    # find whether the node has off-loaded a task
    if recipient:
      departures.append(1)
    else:
      departures.append(0)

    # find how many nodes sent a task to this node
    arrivals.append(sum([1 for idx, recipient in enumerate(recipients) if
                         ((recipient == node_idx) and (nodes[idx].load > 0))]))

  # ----- experience transitions -----
  stop_episode = False
  for node in nodes:
    arr = arrivals[node.idx - 1]
    dep = departures[node.idx - 1]
    recipient = recipients[node.idx - 1]

    underflow, overflow, node_exec, node_off =\
      node.transition(arrivals=arr, departures=dep,
                      executed=executed[node.idx - 1])

    actions[(node.idx - 1)*2] = node_exec
    actions[(node.idx - 1)*2 + 1] = node_off


    # ----- choose appropriate reward -----
    if overflow:
      reward = - payoffs["overflow"]
      stop_episode = True

    elif underflow:
      reward = - payoffs["underflow"]

    else:
      reward = payoffs["alive"]

    # add cost for off-loading
    if recipient:
      transmission_cost = node.off_costs[recipient]
      reward -= transmission_cost

    # add cost for execution
    if node_exec:
      reward -= node.exec_cost

    rewards.append(reward)
    new_states.append(node.load)

  return actions, rewards, new_states, stop_episode


def create_pair(network_type, capacity):
  """ Creates a toy-network of two nodes.
  
  The first node has idx 1 and the second idx 2.

  Args:
    network_type (string): choose among predefined types, which determine all
    characteristics, except for capacity

    capacity (int): capacity of nodes

  Returns:
    a list of Node
  """
  if network_type == "A":
    costs_1 = {0: 0, 2: 2}
    costs_2 = {0: 0, 1: 2}
    serve_cost_1 = 4
    serve_cost_2 = 1
    spawn_rate_1 = 0.5
    spawn_rate_2 = 0.5
    
  else:
    print("Error: this network type has not been defined.")
    quit()

  # ----- build network -----
  nodes = []
  neighbors = [0, 2]
  node = Node(capacity=capacity, neighbors=neighbors, idx=1, off_costs=costs_1,
              exec_cost=serve_cost_1, gen_rate=spawn_rate_1)
  nodes.append(node)
  neighbors = [0, 1]
  node = Node(capacity=capacity, neighbors=neighbors, idx=2, off_costs=costs_2,
              exec_cost=serve_cost_2, gen_rate=spawn_rate_2)
  nodes.append(node)

  return nodes

