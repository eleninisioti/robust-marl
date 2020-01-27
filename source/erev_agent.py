""" Contains the implementation of a reinforcement learning agent."""

import numpy as np
import random
class ErevAgent():


  def __init__(self):
    self.actions = [0,1]
    self.propensities = [1, 1] # should be positive
    self.pure_eq = {0: [], 1: []}

  def decide(self):
    """ Decide whether to attend the bar."""

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

