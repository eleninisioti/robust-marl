""" Contains the implementation of a generic temporal difference learning
agent."""

# ----- generic imports -----
import numpy as np
import random
import itertools
import copy


class Agent:
  """ A base class for temporal difference learning agents.

  An agent always has observability of all nodes of the MAS, as the Q-table
  and policies are learned on the joint state and action space. Depending on
  its type, an agent may have control over the actions of a subset of the
  nodes, so other agents need to communicate their actions. This is useful
  for centralized and joint learners. Independent learners should override
  the __init__ function and define the state-action space accordingly.

  Note: This is an interface, do not create objects of this class.

  Attributes:
    state_space (tuple of int): the dimensions of the state space. each node
    of the MAS is another dimension

    action_space (tuple of int): the dimensions of the action space. each node
    controlled by the agent contributes two dimensions (execute and off-load)

    learn_space (tuple of int): the dimensions of the Q-table

    Qtable (array of float): an array of dimension learn_space

    nodes (list of Node): all nodes of the MAS

    current_state (list of int): of dimension state_space

    current_action (list of int): the action the agent intends to execute in
    the next step

    learn_parameters (dict of str: float): contains the values of different
     hyperparameters including learning rate, discount factor, exploration rate

    log (dict of values): used for logging information for debugging and
     plotting
  """

  def __init__(self, nodes, epsilon, alpha, gamma):
    """ Initializes an agent.

    Args:
      nodes (list of :obj:`Node`): nodes comprising the MAS
      epsilon (float): exploration rate
      alpha (float): learning rate
      gamma (float): discount factor
    """

    # ----- initialize Qtable -----
    self.nodes = nodes

    # define state space
    state_space = []
    for node in nodes:
      state_range = node.capacity + 2
      state_space.append(state_range)  # [0,capacity]
    self.state_space = tuple(state_space)

    # define action space
    action_space = []
    for node in nodes:
      action_space.append(2)  # execute action
      action_space.append(len(node.neighbors))  # off-load actions
    self.action_space = tuple(action_space)

    self.learn_space = self.state_space + self.action_space
    self.Qtable = np.random.uniform(low=0, high=0.001, size=self.learn_space)

    # initialize current state and action
    self.current_state = [0] * len(nodes)  # initial state
    self.current_action = [0, 0] * len(nodes)

    # initialize learning parameters
    self.learn_parameters = {"epsilon": epsilon, "alpha": alpha, "gamma": gamma}

    # initialize dictionary for logging
    self.log = {"updates": []}

  def execute_policy(self, attack_actions, evaluation=False):
    """ Choose the action to perform based on the policy, the exploration
    scheme and the presence of attackers.

    This function is used both during training, when no adversaries can be
    present, and evluation, when no exploration should take place.

    Note: the policy is defined over the action space of defenders. Except
    for minimaxQ, where some nodes are defenders and some opponents,
    all nodes are defenders. Do not confuse adversaries (during evaluation)
    with minimaxQ opponents (during training).

    Args:
       attack_actions (dict of int: int): has the from abs_idx: relative actions
       deployment (bool): indicates whether the policy is currently being
       deployed, in which case no exploration takes place

    Returns:
      a list of actions to execute, where indexes are absolute
    """

    # ----- execute e-greedy policy -----
    x = random.uniform(0, 1)
    if ((x < self.learn_parameters["epsilon"]) and not(evaluation)):

      # perform random move
      self.current_action = []
      for idx, node in enumerate(self.nodes):
        if node in self.control_nodes:
          self.current_action.append(random.randint(0, 1))  # execute action
          self.current_action.append(
            random.randint(0, len(node.neighbors) - 1))

    else:  # perform action based on current policy
      self.onpolicy_action(evaluation)

    # ----- replace actions of attacked nodes -----
    control_abs_idxs = [node.idx for node in self.control_nodes]
    for key, value in attack_actions.items():

      # map to absolute node idx
      abs_key = self.nodes[key].idx

      if abs_key in control_abs_idxs:

        # map to position in current_action
        rel_key = control_abs_idxs.index(abs_key)
        self.current_action[rel_key * 2] = value[0]
        self.current_action[rel_key * 2 + 1] = value[1]

    # map off-load action from relative idxs to absolute idxs
    abs_action = copy.copy(self.current_action)
    for idx, node in enumerate(self.control_nodes):
      neighbors = node.neighbors
      off_action = self.current_action[idx * 2 + 1]
      abs_action[idx * 2 + 1] = neighbors[off_action]

    return abs_action

  def update_qvalue(self, reward, next_state):
    """ Updates a value in the Q-table of the agent using temporal difference
    learning.

    Note: the value of the target policy is defined in sub-classes.

    Args:
      reward (list of float): contains individual node rewards
      next_state (list of int): contains individual node loads
    """

    # get current Qvalue
    current_entry = tuple(self.current_state + self.current_action)
    Qcurrent = self.Qtable[current_entry]

    # compute value of target policy
    target = self.compute_target(next_state)

    # update Q-value
    td_error = sum(reward) + self.learn_parameters["gamma"] * target - Qcurrent
    self.Qtable[current_entry] = Qcurrent +\
                                 self.learn_parameters["alpha"] * td_error

  def update(self, reward, next_state, learn, opponent_action=[]):
    """ Updates an agent after interaction with the environment.
    """
    pass

  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    pass

  def onpolicy_action(self, deployment):
    """ Performs greedy action"""
    pass
