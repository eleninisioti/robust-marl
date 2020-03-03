""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math
import itertools

class QlearningAgent():


  def __init__(self, nodes, K,  prob_attack, robust, epsilon=0.01, alpha=0.1,
               gamma=0.9,
               temperature=0.01):

    # initialize Qtable
    space_shape = []
    for node in nodes:
      state_range = node.capacity
      space_shape.append(state_range)
    for node in nodes:
      action_range = len(node.neighbors)
      space_shape.append(action_range)
    self.space_shape = tuple(space_shape)
    self.Qtable = np.random.random_sample(space_shape)

    # initialize states and actions
    self.nodes = nodes
    nactions = 0
    self.actions = []
    self.current_actions = []
    for node in nodes:
      nactions += (len(node.neighbors)+1)
      self.current_actions.append(0)
      self.actions.extend(node.neighbors)

    self.current_state = [0] * len(nodes)
    self.current_action = [0] * len(nodes)

    # learning parameters
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.temperature = temperature
    self.prob_attack = prob_attack
    self.K = K
    self.robust = robust



  def decide(self, attack, explore):
    """ Decide whether to attend the bar.
    Args:
      attack (int): size of attack

    The presence of attackers  indicates whether we need to calculate
     adversarial actions

    Returns: chosen action
    """
    if attack:
      # calculate adversarial actions
      self.current_actions = self.find_adversarial_actions(self.Qtable[
                                                   self.current_state])
    else:

      # ----- e-greedy -----
      if explore == "egreedy":
        x = random.uniform(0,1)

        if x < self.epsilon:
          self.current_actions = []
          for idx, node in enumerate(self.nodes):
            self.current_actions.append(random.randrange(len(node.neighbors)))
        else:
          current_entry = [slice(None)] * len(self.space_shape)
          for idx, el in enumerate(self.current_state):
            current_entry[idx] = el
          current_entry = tuple(current_entry)
          Qcurrent = self.Qtable[current_entry]
          max_actions_flat = np.argmax(Qcurrent)
          self.current_actions = np.unravel_index(max_actions_flat,
                                                Qcurrent.shape)

      #elif explore == "Boltzmann":
        # TODO

    # map actions to correct nodes
    choices = []
    for idx, node in enumerate(self.nodes):
      neighbors = node.neighbors
      choices.append(neighbors[self.current_actions[idx]])

    return choices

  def find_adversarial_actions(self, Qvalues):
    """
    Finds adversarsial actions for current state and Qtable
    """
    max_actions_flat = np.argmax(Qvalues)
    max_actions = np.unravel_index(max_actions_flat, Qvalues.shape)

    # find all possible subsets of K attackers in N nodes
    indexes = list(range(len(self.nodes)))
    attackers_partitions = list(itertools.combinations(indexes, self.K))

    Qadv = np.max(Qvalues)
    actions = max_actions
    for partition in attackers_partitions:

      # find subset of defenders
      defenders = [node for node in indexes if node not in partition]

      # minimize over attackers
      defend_actions = [el for idx, el in enumerate(max_actions) if idx not in \
                                                            partition]
      indcs = [slice(None)]*len(max_actions)
      for defend in defenders:
        indcs[defend] = max_actions[defend]
      attacker_Qvalues = Qvalues[tuple(indcs)]
      min_actions = np.argmin(attacker_Qvalues)
      min_actions = np.unravel_index(min_actions, attacker_Qvalues.shape)
      # find actions of defenders and attackers
      partition_actions = list(max_actions)
      counter = 0
      for idx, action in enumerate(partition_actions):
        if idx in partition:
          partition_actions[idx] = min_actions[counter]
          counter += 1

      # Qvalue of current partition
      Qvalue = Qvalues[tuple(partition_actions)]
      if Qvalue < Qadv:
        Qadv=Qvalue
        actions = partition_actions
    return actions

  def classical_update(self, new_state, reward):
    """ Performs a classical Q-learning update.
    """
    current_entry = tuple(self.current_state + self.current_action)
    Qcurrent = self.Qtable[current_entry]
    entry = [slice(None)] * len(self.space_shape)
    entry[:len(new_state)] = new_state
    Qnext = self.Qtable[tuple(entry)]
    target = np.max(Qnext)

    self.Qtable[current_entry] = \
      Qcurrent + self.alpha * (sum(reward) + self.gamma * target -
                               Qcurrent)

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



  def update(self, new_state, reward, current_t):
    """ Update propensities.
    """

    # update Qtable
    if self.robust:
      self.robust_update(new_state, reward, current_t=current_t)
    else:
      self.classical_update(new_state, reward)

    # update state
    self.current_state = list(new_state)



