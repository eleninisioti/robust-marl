""" Contains the implementation of a Q-learning agent."""

# ----- generic imports -----
import numpy as np
import random

# ----- project-specific imports -----
from agent import Agent

class QAgent(Agent):
  """ An agent that uses classical Q-learning.

  Attributes:
    control_nodes (list of Node): the nodes whose actions are controlled
  """

  def __init__(self, nodes, epsilon, alpha, gamma):
    """ Initializes a Qlearning agent.
    """

    super().__init__(nodes=nodes, epsilon=epsilon, alpha=alpha, gamma=gamma)

    self.control_nodes = self.nodes

    # initialize policy
    self.policies = []
    for node in self.control_nodes:
      neighbors = node.neighbors
      policy_action_space = [2, len(neighbors)]
      policy_space = self.state_space + tuple(policy_action_space)
      self.policies.append(np.ones(shape=policy_space) / np.sum(
        policy_action_space))


  def update(self, reward, next_state, def_action, opponent_action, learn=True):
    """ Updates an agent after interaction with the environment.

    Args:
      reward (list of float): contains individual node rewards
      next_state (list of int): contains individual node loads
      learn (bool): indicates whether the Q-table will be updated
    """
    if learn:
      self.update_qvalue(reward=reward, next_state=next_state,
                         def_action=def_action)

    self.current_state = next_state

  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.

    For Q-learning, this is defined as max_a Q(s,a).

    Args:
      next_state (list): contains the state of each node in the MAS
      
    Returns: the float value of the target policy
    """
    # get Qvalues for current state
    entry = [slice(None)] * len(self.state_space)
    entry[:len(next_state)] = next_state
    qnext = self.Qtable[tuple(entry)]

    # pick Q-value of greedy action
    return np.max(qnext)

  def onpolicy_action(self):
    """ Finds the greedy action based on the deterministic policy.
    """
    # get q-values for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    qcurrent = self.Qtable[tuple(current_entry)]

    # find greedy action
    max_actions_flat = np.argmax(qcurrent)
    all_indices = list(zip(*np.where(qcurrent == np.max(qcurrent))))
    random_ind = random.randint(0,len(all_indices)-1)
    indices = all_indices[random_ind]
    self.current_action = list(indices)

    # ----- compute deterministic policies for each control node -----
    # set all entries to zero
    for idx, policy in enumerate(self.policies):
      policy[tuple(current_entry)] = np.zeros(policy[tuple(
        current_entry)].shape)
      self.policies[idx] = policy

    # set entries for greedy action to 1
    for idx, action in enumerate(self.current_action[::2]):
      comb_action = (action, self.current_action[idx*2+1])
      self.policies[idx][tuple(current_entry)][comb_action] = 1