""" Contains the implementation of the bar in the El Farol Bar Problem.

We currently assume that there are multiple agents and a single bar.
"""

class Bar:
  """ A bar has a certain capacity of agents. If the number of agents
  attending exceed this capacity, then all agents ..."""

  attendants = 0

  def __init__(self, capacity=60, technique="Erev"):
    self.capacity = capacity
    self.technique = technique

  def visit(self):
    """ An agent visits the bar"""
    self.attendants += 1

  def payoff(self, action, turnout):
    """ Calculate payoff to agent.

    The El Farol bar problem is a repeated game described by a 2d payoff
    matrix that has the following variables: G, the payoff of going to the
    bar when it is uncrowded, B the payoff of going to the bar when it is
    crowded and S, the payoff when the agent does not go to the bar.
    G > S > B.
    """
    G = 5
    S = 3
    B = 1
    if action==0:
      return S
    else:
      if turnout > self.capacity:
        return B
      else:
        return G




  def social_welfare(self, turnout):
    """ Quantify social welfare based on turnout.
    """
    # if turnout > self.capacity:
    #   social_welfare = 0
    # else:
    #   social_welfare = 1
    social_welfare = turnout
    return social_welfare

  def reward(self, action, turnout):
    """ Returns the reward to an agent, depending on its technique.
    """
    if self.technique == "Erev":
      return self.payoff(action, turnout)
    elif self.technique == "Arthur":
      return self.social_welfare(turnout)