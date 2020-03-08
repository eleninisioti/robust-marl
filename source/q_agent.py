""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math
import itertools

from agent import *

class QAgent(Agent):


  def __init__(self, nodes, K,  prob_attack, robust, epsilon=0.01, alpha=0.01,
               gamma=0.7, temperature=0.01):

    super().__init__(nodes, K,  prob_attack, robust, epsilon, alpha,
               gamma, temperature)

  def update(self, reward, next_state, learn=True):
    """ Updates an agent after interaction with the environment.
    """
    if learn:
      self.update_Qvalue(reward=reward, next_state=next_state)
    self.current_state = next_state

  def compute_target(self, Qtable):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    return np.max(Qtable)


  def execute_policy(self, attack_actions, exploration=True):
    """ Choose the action to perform based on the policy.
    """
    return self.perform_action(attack_actions, exploration)
