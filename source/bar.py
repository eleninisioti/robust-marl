""" Contains the implementation of the bar in the El Farol Bar Problem.

We currently assume that there are multiple agents and a single bar.
"""

class Bar:
  """ A bar has a certain capacity of agents. If the number of agents
  attending exceed this capacity, then all agents ..."""

  attendants = 0

  def __init__(self, capacity=60):
    self.capacity = capacity

  def visit(self):
    """ An agent visits the bar"""
    self.attendants += 1

  def social_welfare(self, turnout):
    """ Quantify social welfare based on turnout.
    """
    # if turnout > self.capacity:
    #   social_welfare = 0
    # else:
    #   social_welfare = 1
    social_welfare = turnout
    return social_welfare