""" Contains the implementation of an agent in the El Farol Bar Problem.

We currently assume that there are multiple agents and a single bar.
"""
import numpy as np
import random

from arthur_model import *

class Agent:

  def __init__(self, id, bar, nagents, horizon=4, K=16):
    self.id = id
    self.info = {} # contains all information available to the agent
    random.seed(id)
    self.history = [random.randint(0,bar.capacity)]*horizon # contains the
    # history of states
    # used for
    # learning
    self.horizon = horizon
    self.models = []
    self.bar = bar

    # TODO: initialize models
    # pick a subset of all possible seeds
    random.seed(id)
    seeds = random.choices(list(range(nagents*100)), k=K)
    for k in range(0,K):
      self.models.append(ArthurModel(seed=seeds[k], bar=bar, nagents=nagents))

    self.active_predictor = 0 # the index of the best model


  def decide(self):
    """ Decide whether to attend a bar."""

    # consult best model
    attend = self.models[self.active_predictor].consult(self.history)
    return attend

  def update(self, turnout):
    """ Update agent's info and history.

    Returns a binary value, indicating whether the agent is optimistic.
    """
    # TODO: should I update all models?
    for model in self.models:
      model.consult(self.history)
      model.update(turnout)


    if len(self.history) == self.horizon:
      self.history.pop(0)
    self.history.append(turnout)

    # update model
    self.models[self.active_predictor].update(turnout)

    # choose model with best score as active predictor
    self.active_predictor = np.argmax([model.score for model in self.models])

    # keep last prediction
    if self.models[self.active_predictor].last_predict < self.bar.capacity:
      return 1
    else:
      return 0


