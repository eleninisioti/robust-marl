""" Contains the implementation of a prediction model.

"""
import numpy as np

class Model:

  def __init__(self, bar, rate=0.9):
    self.score = 0
    self.rate = rate
    self.last_predict = 0
    self.bar = bar

  def predict(self):
    pass


  def update(self, turnout):
    self.score = self.rate * self.score + (1 - self.rate) * 1 / (np.abs(
      self.last_predict - turnout) + 1)

  def consult(self):
    """ Consult a model on whether you should attend the bar.
    """
    pass

