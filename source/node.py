""" Contains the implementation of a network node. """

# ----- generic imports -----
import numpy as np
import random

class Node:

  """ A network node that has the capability of executing and off-loading
  tasks to its neighbors.

  Attributes:
    capacity (int): maximum number of packets for which an overflow does not
    occur
    idx (int): a unique positive integer identifier for this node
    cutoff (int): minimum number of packets for which an underflow does not
     occur
    gen_rate (float): probability of generation of new packet
    exec_cost (int): cost for serving a packet
    neighbors (list of int): the idxs of neighboring nodes
    load (int): current number of tasks in node
    off_costs (list of float): each element is the cost of off-loading to
    the corresponding neighbor

  """

  def __init__(self, capacity, idx, off_costs, gen_rate, exec_cost, neighbors,
               cutoff=0):

    """ Initializes a network node.

    Args:
      capacity (int): maximum number of packets for which an overflow does
      not occur
      idx (int): a unique positive integer identifier for this node
      cutoff (int): minimum number of packets for which an underflow does not
       occur
      gen_rate (float): probability of generation of new packet
      exec_cost (int): cost for serving a packet
      neighbors (list of int): the idxs of neighboring nodes
      load (int): current number of tasks in node
      off_costs (dict of int: float): each element is the cost of off-loading to
      the corresponding neighbor
    """

    # initialize node characteristics
    self.capacity = capacity
    self.cutoff = cutoff
    self.gen_rate = gen_rate
    self.off_costs = off_costs
    self.exec_cost = exec_cost
    self.neighbors = neighbors
    self.idx = idx
    self.load = 0

    # initialize structure with statistics for debugging
    self.statistics = {"generations": [], "executions": 0, "departures": 0,
                       "arrivals": 0}

  def reset(self):
    """ Reset the load of the node.
    """
    self.load = 0

  def transition(self, arrivals, departures, executed):
    """ Implements the state transition of a node.

    The state is identical to the load of the node.

    Args: arrivals (int): number of packets arriving from neighboring nodes
          departures (int): number of packets sent to neighboring nodes
          executed (int): number of packets the node chose to serve

    Returns:
      a boolean indicating underflow, a boolean indicating overflow and the
      number of tasks executed when we don't allow negative loads
    """
    underflow = False
    overflow = False

    # a new task is generated
    spawned = np.random.binomial(n=1, p=self.gen_rate)

    # if there's not enough tasks to execute node's actions, we first ignore
    # the action to execute, and then the action to off-load
    if (self.load - departures - executed) < 0:
      executed = 0

      if (self.load - departures) < 0:
        departures = 0

    # transition
    self.load = self.load - departures - executed + spawned + arrivals

    # detect underflow
    if self.load < self.cutoff:
      underflow = True
      self.load = self.cutoff

    # detect overflow
    if self.load > self.capacity:
      overflow = True
      self.load = self.capacity + 1

    # update statistics about node
    if hasattr(self, 'statistics'):
      self.statistics["generations"].append(spawned)
      self.statistics["executions"] += executed
      self.statistics["departures"] += departures
      self.statistics["arrivals"] += arrivals
    else:
      self.statistics = {"generations": [], "executions": 0, "departures": 0,
                       "arrivals": 0}

    return underflow, overflow, executed, departures





