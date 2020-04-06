""" Contains the implementation of a Sarsa agent."""

import copy
import numpy as np

from q_agent import *

class DoubleAgent(QAgent):
  """ An agent that uses classical SARSA.

  This class uses the QAgent implementation and simply changes the target
  policy.
  """

  def __init__(self, nodes, adjust_parameters, epsilon=0, alpha=0.1, gamma=1.0,
               temperature=0.01):

    super().__init__(nodes, adjust_parameters, epsilon=0, alpha=0.1, gamma=1.0,
               temperature=0.01)

    self.Qtable_b = copy.copy(self.Qtable)


  def compute_target(self, next_state, updateA):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    entry = [slice(None)] * len(self.state_space)
    entry[:len(next_state)] = next_state

    Qb = self.Qtable_b[tuple(entry)]
    Qa = self.Qtable[tuple(entry)]
    if updateA:

      max_actions_flat = np.argmax(Qa)
      max_actions = list(np.unravel_index(max_actions_flat,
                                                  Qa.shape))
      Qnext = Qb[tuple(max_actions)]

    else:
      max_actions_flat = np.argmax(Qb)
      max_actions = list(np.unravel_index(max_actions_flat,
                                                  Qb.shape))
      Qnext = Qa[tuple(max_actions)]

    return Qnext


  def update_Qvalue(self, reward, next_state):
    """ Updates the Qvalue function of the agent using temporal difference
    learning.

    Args:
      reward (list of float): one-dimensional, contains individual node rewards
      next_state (list of int): one-dimensional, contains individual node loads

      """

    current_entry = tuple(self.current_state + self.current_action)

    # update based on A table
    if np.random.rand() < 0.5:
      print("updating A")
      Qcurrent = self.Qtable[current_entry]

      target = self.compute_target(next_state, updateA=True) # needs to
      # calculate
      # QB[s,
      # argmaxQa]

      self.Qtable[current_entry] = Qcurrent + \
                                   self.alpha * (sum(
        reward) + self.gamma * target - Qcurrent)

    # update based on B table
    else:
      print("updating B")
      Qcurrent = self.Qtable_b[current_entry]

      target = self.compute_target(next_state, updateA=False) # needs to
      # calculate
      # QB[s,
      # argmaxQa]

      self.Qtable_b[current_entry] = Qcurrent + \
                                   self.alpha * (sum(
        reward) + self.gamma * target - Qcurrent)






