""" Contains the implementation of a generic temporal difference learning
agent."""

import numpy as np
import random
import itertools
import copy

class Agent():
  """ A base class for temporal difference learning agents.

  Note: Do not create objects of this class.

  The state and action space of an agent contains information only about the
  nodes assigned to that agent. Thus, using this class, we can implement
  a centralized learner or independent learners or coordinated learners,
  where each learner find a policy over its own state-action space. If we
  have joint Q-learners, the Qvalue function remains the same, but the policy of learners
  is defined only over their own action space (not the joint one).
  """

  def __init__(self, nodes, epsilon, alpha, gamma, temperature,
               explore="egreedy"):
    """ Initializes a generic agent.

    Args:
      nodes (list of :obj:`Node`): a list of nodes that the agent is
      controlling
      epsilon (float): exploration rate
      alpha (float): learning rate
      gamma (float): discount factor
      temperature (float): temperature used for Boltzmann exploration
    """

    # ----- initialize Qtable -----
    # define state space
    self.state_space = []
    for node in nodes:
      state_range = node.capacity + 2
      self.state_space.append(state_range) # [0,capacity]

    # define action space
    self.action_space = []
    for node in nodes:
      self.action_space.append(2) # serve action
      self.action_space.append(len(node.neighbors)) # send action

    self.learn_space = self.state_space + self.action_space
    self.Qtable = np.random.uniform(low=0, high=0.001, size=tuple(
      self.learn_space))

    # set entries of terminal states to 0
    self.updates = []


    # initialize current state and action
    self.nodes = nodes
    self.current_state = [0] * len(nodes) # no load
    self.current_action = [0,0] * len(nodes)

    # initialize learning parameters
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.temperature = temperature
    self.explore = explore



  def find_adversarial_actions(self, K):
    """ Finds adversarial actions for current policy.

    An adversarial attack consists in attackers choosing the K nodes to
    attack so that the Q-table of the minimax game played by adversaries and
    players has the minimum value.

    Args:
      K (int): number of attackers

    Returns:
      the indexes of adversaries and an adversarial action that is a
      dictionary with entries (adv_index): adv_action
    """
    # get Qtable at current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    Qcurrent = self.Qtable[tuple(current_entry)]

    # find greedy actions
    max_actions_flat = np.argmax(Qcurrent)
    max_actions = np.unravel_index(max_actions_flat, Qcurrent.shape)

    # find all possible subsets of K attackers in N nodes
    indexes = list(range(len(self.nodes)))
    attackers_partitions = list(itertools.combinations(indexes, K))


    # ----- find partition with minimum Q-value -----

    # initialize search variables
    Qadv = np.max(Qcurrent)
    actions = max_actions
    worst_partition = attackers_partitions[0]
    for partition in attackers_partitions:

      # find subset of defenders
      defenders = [node for node in indexes if node not in partition]

      # find subset of Qtable when defenders are maximizing
      indcs = [slice(None)] * len(max_actions)
      for defend in defenders:
        defend_ind = defend*2 # cause each node has two actions
        indcs[defend_ind] = max_actions[defend_ind]
        indcs[defend_ind+1] = max_actions[defend_ind+1]
      defend_Qvalues = Qcurrent[tuple(indcs)]

      # minimize remaining actions over attackers
      min_actions = np.argmin(defend_Qvalues)
      min_actions = np.unravel_index(min_actions, defend_Qvalues.shape)

      # consolidate in single list
      partition_actions = list(max_actions)
      counter = 0
      for idx, action in enumerate(partition_actions):
        if idx in partition:
          partition_actions[idx] = min_actions[counter]
          counter += 1

      # keep partition with minimum Qvalue
      Qvalue = Qcurrent[tuple(partition_actions)]
      if Qvalue < Qadv:
        Qadv = Qvalue
        actions = partition_actions
        worst_partition = partition

    # get (absolute) indexes of adversaries and their actions
    adv_actions = {}
    for idx in worst_partition:
      trans_idx = idx*2
      absolute_idx = self.nodes[idx].idx
      serve_action = actions[trans_idx]
      send_action = actions[trans_idx + 1]
      adv_actions[absolute_idx] = [serve_action, send_action]
    return adv_actions

  def execute_policy(self, attack_actions, exploration=True):
    """ Choose the action to perform based on the policy, the exploration
    scheme and the presence of attackers.

    This function is used both during training, when no adversaries can be
    preset, and testing, when no exploration should take place.

    Note: the policy is defined over the action space of defenders. Except
    for minimaxQ, where some nodes are defenders and some opponents,
    all nodes are defenders. Do not confuse attackers with minimaxQ opponents.

    Args:
       attack_actions (list of int): one-dimensional list of N*2
       exploration (bool): indicates whether exploration should take place
    ."""
    # ----- e-greedy -----
    if self.explore == "egreedy":

      x = random.uniform(0, 1)
      if ((x < self.epsilon) and exploration): # random move
        self.current_action = []
        for idx, node in enumerate(self.nodes):
          if (node.idx) in self.defenders:
            self.current_action.append(random.randint(0,1))
            self.current_action.append(random.randint(0,
            len(node.neighbors)-1))
            #self.current_action.append(0)
            #self.current_action.append(0)

      else: # greedy move
        self.greedy_action(just_test=(not exploration))

    # ----- implement attack -----
    # find if attacked nodes are controlled by this agent and map absolute
    # indexes to action indexes
    # attacked_nodes = []
    # nodes_idxs = [node.idx for node in self.nodes]
    # keep_nodes = []
    # for key, value in attack_actions.items():
    #   for pos, node_idx in enumerate(nodes_idxs):
    #     if key == node_idx:
    #       keep_nodes.append(pos)
    #       attacked_nodes.append(key)
    #
    # for pos, node in enumerate(attacked_nodes):
    #   value = attack_actions[node]
    #   self.current_action[keep_nodes[pos]] = value[0]
    #   self.current_action[keep_nodes[pos] + 1] = value[1]

    if len(self.current_action) > 3:

      for key, value in attack_actions.items():
        self.current_action[key*2] = value[0]
        self.current_action[key*2 + 1] = value[1]
    else:
      for key, value in attack_actions.items():

        self.current_action[0] = value[0]
        self.current_action[1] = value[1]

    # map send action to absolute idxs
    transformed_action = copy.copy(self.current_action)
    for idx, node in enumerate(self.control_nodes):
      neighbors = node.neighbors
      send_action = self.current_action[idx*2 + 1]
      transformed_action[idx*2+1] = neighbors[send_action]

    return transformed_action


  def update_Qvalue(self, reward, next_state):
    """ Updates the Qvalue function of the agent using temporal difference
    learning.

    Args:
      reward (list of float): one-dimensional, contains individual node rewards
      next_state (list of int): one-dimensional, contains individual node loads

      """

    current_entry = tuple(self.current_state + self.current_action)

    Qcurrent = self.Qtable[current_entry]

    target = self.compute_target(next_state)

    self.Qtable[current_entry] = Qcurrent +\
      self.alpha * (sum(reward) + self.gamma * target - Qcurrent)

    #self.updates.append(Qcurrent -self.Qtable[current_entry])

  def update(self):
    """ Updates an agent after interaction with the environment.
    """
    pass

  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    pass

  def greedy_action(self):
    """ Performs greedy action"""
    pass









