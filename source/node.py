""" Contains the implementation of a network node. """

import numpy as np

class Node:

  def __init__(self, capacity, quick_node, idx, costs, serve_cost, neighbors,
               cutoff=0):
    self.capacity = capacity
    self.cutoff = cutoff
    if quick_node:
      self.spawn_rate = 0.2
      #self.serve_rate = 2* self.spawn_rate
    else:
      if idx ==2:
        self.spawn_rate = 0.5
        #self.serve_rate = 0
      else:
        self.spawn_rate = 0.9
        #self.serve_rate = 0
    self.load = 0
    self.neighbors = neighbors
    self.idx = idx
    self.costs = costs
    self.serve_cost = serve_cost

  def reset(self):
    """ Reset node to start new episode.

    Resets the load
    """
    self.load = 0


  def transition(self, arrivals, departures, served):
    """ Implements the state transition of a node.

    The state equals the load of the node.

    Args: arrivals (int): number of packets arriving from neighboring nodes
          departures (int): number of packets sent to neighboring nodes

    """
    underflow = False
    overflow = False

    self.load = self.load - departures - served
    if self.load < self.cutoff:
      underflow = True
      self.load = self.cutoff

    # a new packet is spawned
    spawned = np.random.binomial(n=1, p=self.spawn_rate)

    # an existing packet is served (removed from the new formulation)
    #served = np.random.binomial(n=1, p=self.serve_rate)

    self.load = self.load + spawned + arrivals

     # what happens if negative load?

    if self.load > (self.capacity-1):
      overflow = True
      self.load = self.capacity

    return underflow, overflow





