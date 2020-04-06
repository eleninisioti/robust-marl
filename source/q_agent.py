""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math
import itertools

from agent import *

class QAgent(Agent):
  """ An agent that uses classical Q-learning.
  """


  def __init__(self, nodes, adjust_parameters, epsilon=0, alpha=0.1, gamma=0.9,
               temperature=0.01):
    """ Initializes a Qlearning agent.

    N nodes are assigned to the agent for control.

    Args:
      nodes (list of :obj:`Node`): a list of nodes that the agent is
      controlling
      epsilon (float): exploration rate
      alpha (float): learning rate
      gamma (float): discount factor
      temperature (float): temperature used for Boltzmann exploration
      adjust_parameters (bool): indicates whether the learning and
      exploration rate will be decreased at each iteration
    """

    super().__init__(nodes=nodes, epsilon=epsilon, alpha=alpha, gamma=gamma,
                     temperature=temperature)
    self.adjust_parameters = adjust_parameters
    self.control_nodes = self.nodes

    # none of the nodes is an adversary
    self.defenders =  [node.idx for node in self.nodes]

  def update(self, reward, next_state, learn=True):
    """ Updates an agent after interaction with the environment.

    During learning both the Q-table and the state are updating. Durinng
    testing only the state is updated
    """
    if learn:
      self.update_Qvalue(reward=reward, next_state=next_state)
      if self.adjust_parameters:
        self.alpha = self.alpha * 0.8
        self.epsilon = self.epsilon * 0.8
    self.current_state = next_state


  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.

    For Q-learning, this is defined as max_a Q(s,a).

    Args:
      next_state (list): one-dimensional, contains the state of each node
      assigned to the agent
     """
    # isolate Qtable for current state
    entry = [slice(None)] * len(self.state_space)
    entry[:len(next_state)] = next_state

    Qnext = self.Qtable[tuple(entry)]
    return np.max(Qnext)



  def greedy_action(self, just_test=False):
    """ Finds the greedy action based on the deterministic policy.
    """

    # isolate Qtable for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    Qcurrent = self.Qtable[tuple(current_entry)]

    # find greedy action
    max_actions_flat = np.argmax(Qcurrent)
    self.current_action = list(np.unravel_index(max_actions_flat,
                                               Qcurrent.shape))

    # compute deterministic policies for each action column
    self.policies = []
    for node in self.control_nodes:
      node_action_space = [2]
      node_action_space.extend([len(node.neighbors)])
      self.policies.append(np.zeros(tuple(node_action_space)))

    for idx, action in enumerate(self.current_action[::2]):
      node_idx = idx
      self.policies[node_idx][action] = 1
      self.policies[node_idx][self.current_action[idx+1]] = 1
