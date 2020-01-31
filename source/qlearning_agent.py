""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math

class QlearningAgent():


  def __init__(self, min_payoff, epsilon=0.01, alpha=0.1, gamma=0,
               temperature=0.1):
    self.min_payoff = min_payoff
    self.actions = [0,1]
    self.qvalues = [0, 0] # should be nonnegative
    self.pure_eq = {0: [], 1: []}
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.temperature = temperature

  def attack(self, force_go=True, reverse=False):
    """ An adversary attacks the agent.

    Args:
      force_go (bool): if True, force an agent to go, if force, force to stay
      reverse (bool); if True, reverse the propensities of the agent
    """
    if reverse:
      # TODO: not sure if this type of attack is useful
      temp = self.propensities[0]
      self.propensities[0] = self.propensities[1]
      self.propensities[1] = temp
    else:
      self.attacked = True
      if force_go:
        self.forced_action = 1
      else:
        self.forced_action = 0


  def decide(self, attacked, explore="Boltzmann"):
    """ Decide whether to attend the bar.
    Args:
      attacked (bool): if True, agent is under attack

    Returns: chosen action
    """
    if attacked:
      self.current_action = self.forced_action
      self.pure_eq[self.forced_action].append(1)
      self.pure_eq[np.abs(1-self.forced_action)].append(0)

    else:

      # ----- e-greedy -----
      if explore == "egreedy":
        x = random.uniform(0,1)

        if x < self.epsilon:
          self.current_action = random.choice(self.actions)
        else:

          # break ties arbitrarily
          if np.abs(self.qvalues[0]-self.qvalues[1]) < 0.00000001:
            self.current_action = random.choice(self.actions)
          else:
            self.current_action = np.argmax(self.qvalues)
      elif explore == "Boltzmann":
        if self.temperature == 0:
          self.current_action = np.argmax(self.qvalues)
        else:
          probs = []
          for action in self.actions:
            probs.append(math.exp(self.qvalues[action]/self.temperature))
          probs = [prob/sum(probs) for prob in probs]
          x = random.uniform(0,1)
          if x < probs[0]:
            self.current_action = 0
          else:
            self.current_action = 1
      self.pure_eq[self.current_action].append(1)
      self.pure_eq[np.abs(1 - self.current_action)].append(0)

    attend = [False,True][self.current_action]
    return attend

  def update(self, payoff, attacked):
    """ Update propensities.
    """
    if not attacked:
      reward = payoff - self.min_payoff
      self.qvalues[self.current_action] =\
        (1-self.alpha)*self.qvalues[self.current_action] + \
        self.alpha*(reward + self.gamma*np.max(self.qvalues))
    return self.current_action

