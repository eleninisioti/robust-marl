""" Contains the implementation of a prediction model as described
 in https://ocw.tudelft.nl/wp-content/uploads/ElFarolArtur1994.pdf and
 further analyzed in https://pdfs.semanticscholar.org/7c7b/35114b3119ef47bb0d59cae6c6569ad63cc3.pdf.

A model is a mapping from histories to turnouts, generated randomly.
"""

import numpy as np
import random
from model import *

class ArthurModel(Model):

  def __init__(self, bar, seed, nagents, rate=0.9):
    self.score = 0
    self.rate = rate
    self.last_predict = 0
    self.bar = bar
    self.seed = seed
    self.nagents = nagents


  def predict(self, history):
    """ Predicts turnout.
    """
    rvalues = []
    turnout = 0

    # get a prediction that depends on the history values
    for h in history:
      random.seed(self.seed+h) # a unique seed and the model is self-consistent
      turnout += random.uniform(0, self.nagents)
    turnout = int(np.round(turnout/len(history)))
    self.last_predict = turnout
    return turnout


  def consult(self, history):
    """ Consult a model on whether you should attend the bar.
    """

    # predict turnout
    turnout = self.predict(history)

    if turnout <= self.bar.capacity:
      attend = True
    else:
      attend = False

    return attend

