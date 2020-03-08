""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math
import itertools

from agent import *

class RomQAgent(Agent):


  def __init__(self, nodes, K,  prob_attack, robust, epsilon=0.01, alpha=0.1,
               gamma=0.9, temperature=0.01):

    super().__init(nodes, K,  prob_attack, robust, epsilon, alpha,
               gamma, temperature)




  def update(self, reward, next_state):
    """ Updates an agent after interaction with the environment.
    """
    self.update_Qvalue(reward=reward, next_state=next_state)



  def compute_target(self, Qstate):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """

    actions = self.find_adversarial_actions(Qstate)
    Vtarget = Qstate[tuple(actions)]


    return Vtarget


  def execute_policy(self, attack_actions):
    """ Choose the action to perform based on the policy.
    """
    return self.perform_action(attack_actions)

  def robust_update(self, new_state, reward, current_t):
    """ Our robust Q-learning. """

    # subset Qtable based on new state
    # TODO: desperately needs refactoring to work with list

    entry =  [slice(None)]*len(self.space_shape)
    entry[:len(new_state)] = new_state
    Qnext = self.Qtable[tuple(entry)]

    # ----- find adversarial Qvalue ------
    actions = self.find_adversarial_actions(Qnext)
    Qadv = Qnext[tuple(actions)]

    # find cooperative Qvalue
    Qcoop = np.max(self.Qtable[new_state])

    # find prob_attack
    prob_under_attack = self.prob_attack*current_t
    target = (prob_under_attack)*Qadv + (1-prob_under_attack)*Qcoop
    current_entry = tuple(self.current_state+ self.current_action)
    Qcurrent = self.Qtable[current_entry]
    self.Qtable[current_entry] =\
      Qcurrent +  self.alpha*(sum(reward) +  self.gamma*target -
                                      Qcurrent)



  def update(self, next_state, reward):
    """ Update propensities.
    """
    self.update_Qvalue(reward=reward, next_state=next_state)





