""" Contains the implementation of a reinforcement learning agent."""

import numpy as np
import random
class ErevAgent():


  def __init__(self):
    self.actions = [0,1]
    self.propensities = [1, 1] # should be positive
    self.pure_eq = {0: [], 1: []}

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


  def decide(self, attacked):
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
      # pick action with maximum propensity
      probs = []
      for a in self.actions:
        prob = self.propensities[a] / sum(self.propensities)
        probs.append(prob)
        if prob > 0.9:
          self.pure_eq[a].append(1)
        else:
          self.pure_eq[a].append(0)

      # random sample
      x = random.uniform(0,1)
      if x < probs[0]:
        self.current_action = 0
      else:
        self.current_action = 1
    attend = [False,True][self.current_action]
    return attend

  def update(self, payoff):
    """ Update propensities.
    """
    self.propensities[self.current_action] = self.propensities[
                                            self.current_action] + payoff
    return self.current_action

