""" Contains the implementation of a MinimaxQ-learning agent."""

# ----- generic imports -----
import numpy as np
import random
import math
import itertools

# ----- project-specific imports -----
from agent import *
from tools import *


class MinimaxQAgent(Agent):
  """ An agent that uses minimax-Q.

  Nodes are divided into defenders and opponents, and a policy is learned over
  the joint action space of defenders. By creating a different MinimaxQAgent
  for each node, where all others are opponents, we can learn a different
  policy for each node.

  Attributes:
    control_nodes (list of Node): the nodes whose actions are controlled
    determ_execution (bool): indicates whether execution of policies during
       deployment should be deterministic
    V (array of float): an array of dimension state_space
  """

  def __init__(self, nodes, epsilon, alpha, gamma, opp_idxs,
               determ_execution):
    """ Initialize minimax-Q agent.

    Args:
      opp_idxs (list of int): absolute indexes of opponents
      determ_execution (bool): indicates whether execution of policies during
       deployment should be deterministic
    """

    super().__init__(nodes=nodes, epsilon=epsilon, alpha=alpha, gamma=gamma)

    self.determ_execution = determ_execution
    
    # determine control nodes
    idxs = [node.idx for node in nodes]
    defend_idxs = [idx for idx in idxs if idx not in opp_idxs]
    self.control_nodes = [node for node in nodes if node.idx in defend_idxs]
    
    # initialize policy
    self.policies = []
    for node in self.control_nodes:
      neighbors = node.neighbors
      policy_action_space = [2, len(neighbors)]
      policy_space = self.state_space + tuple(policy_action_space)
      self.policies.append(np.ones(shape=policy_space) / np.sum(
        policy_action_space))

    # initialize V-table
    self.V = np.random.uniform(low=0, high=0.0001, size=tuple(self.state_space))


  def update(self, reward, next_state, opponent_action, learn=True):
    """ Updates an agent after interaction with the environment.

    Args:
      reward (list of float): contains individual node rewards
      next_state (list of int): contains individual node loads
      learn (bool): indicates whether the Q-table will be updated
    """

    # ----- map opponent action to relative idxs -----
    trans_action = opponent_action
    opp_idxs = [node.idx for node in self.nodes if node not in
                                                  self.control_nodes]

    for idx, action in enumerate(opponent_action):
      if idx%2 != 0: # only off-loading actions require mapping

        # get opponent absolute idx
        opp_idx = opp_idxs[int(idx/2)]

        # get  opponent node
        opp_node = [node for node in self.nodes if node.idx == opp_idx][0]

        # get its neighbors
        opp_neighbs = opp_node.neighbors

        # get relative index of neighbor opponent chose to off-load to
        position = opp_neighbs.index(action)

        trans_action[idx] = position

    # position actions based on partition in control nodes and opponents
    comb_action = []
    count_control = 0
    count_adv = 0
    for node in self.nodes:
      if node in self.control_nodes:
        comb_action.append(self.current_action[count_control])
        comb_action.append(self.current_action[count_control+1])
        count_control += 1
      else:
        comb_action.append(trans_action[count_adv])
        comb_action.append(trans_action[count_adv + 1])
        count_adv += 1
    self.current_action = comb_action

    if learn:
      self.update_qvalue(reward=reward, next_state=next_state)
      self.update_policy()
    self.current_state = next_state

  def update_policy(self, retry=False):
    """ Update the policy and corresponding value function of agent.

    The update requires the solution of a linear program.

    Args:
      retry (bool): indicates if we'll try to solve the LP for a second time
    """

    # get q-values for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    qtable = self.Qtable[tuple(current_entry)]

    num_a = len(self.control_nodes) * 4
    num_o = (len(self.nodes) - len(self.control_nodes)) * 4
    
    # ----- swap axes in Q-table so that adversaries are first -----
    map = {}
    count = 0
    opp_nodes = [node for node in self.nodes if node not in self.control_nodes]
    for opp in opp_nodes:
      pos = self.nodes.index(opp)
      map[count] = pos
      map[count + 1] = pos + 1
      count += 1

    for key, value in map.items():
      qtable = np.swapaxes(qtable, key, value)

    qtable = np.reshape(qtable, (num_o, num_a))
    
    # solve linear program
    res = solve_LP(num_a, num_o, qtable)

    if res.success:
      current_pi = self.policies[0][tuple(current_entry)]
      self.policies[0][tuple(current_entry)] = np.reshape(res.x[1:],
                                                     current_pi.shape)
      self.V[tuple(current_entry)] = res.x[0]

    elif not retry:
      return self.update_policy(retry=True)

    else:
      print("Alert : %s" % res.message)


  def compute_target(self, next_state):
    """ Computes the value of the target policy.

    Args:
      next_state (list of int): contains individual node loads
    """
    return self.V[tuple(next_state)]


  def onpolicy_action(self, deployment):
    """ Performs greedy action
    """
    # get state-specific policy
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    policy = self.policies[0][tuple(current_entry)]

    if self.determ_execution and deployment:
      # ----- execute deterministic policy during deployment -----
      # get q-values for current state
      current_entry = [slice(None)] * len(self.state_space)
      for idx, el in enumerate(self.current_state):
        current_entry[idx] = el
      qcurrent = self.Qtable[tuple(current_entry)]

      # find greedy action
      max_actions_flat = np.argmax(qcurrent)
      current_action = list(np.unravel_index(max_actions_flat,
                                             qcurrent.shape))

      # find dimensions of defenders
      defender_dims = []
      for idx, node in enumerate(self.nodes):
        if node in self.control_nodes:
          defender_dims.extend([idx*2,idx*2+1])

      # isolate defender actions
      defend_action = []
      for node_idx, action in enumerate(current_action):
        if node_idx in defender_dims:
          defend_action.append(action)

      self.current_action = defend_action

    else:
      rand = np.random.rand()
      flat_pi = np.ndarray.flatten(policy)
      cumSumProb = np.cumsum(flat_pi)

      action = 0
      while rand > cumSumProb[action]:
        action += 1
      self.current_action = list(np.unravel_index(action, policy.shape))

    return self.current_action
