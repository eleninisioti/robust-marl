""" Contains the implementation of a Sarsa agent."""


from q_agent import *

class SarsaAgent(QAgent):
  """ An agent that uses classical SARSA.

  This class uses the QAgent implementation and simply changes the target
  policy.
  """


  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    Qnext = self.Qtable[tuple(next_state+ self.current_action)]
    return Qnext





