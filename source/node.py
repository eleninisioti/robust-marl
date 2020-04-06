""" Contains the implementation of a network node. """

import numpy as np

class Node:

  def __init__(self, capacity, idx, costs, spawn_rate, serve_cost, neighbors,
               cutoff=0):

    """ Initializes a network node.

    Args:
      capacity (int): maximum number of packets for which an overflow does
      not occur
      idx (int): a unique positive integer identifier for this node
      cutoff (int): minimum number of packets for which an underflow does not
       occur
      spawn_rate (float): between 0 and 1, indicates probability of attack in a
       time step
      serve_cost (int): cost for serving a packet
      neighbors (list of int): the idxs of neighboring nodes
    """

    # initialize node characteristics
    self.capacity = capacity
    self.cutoff = cutoff
    self.spawn_rate = spawn_rate
    self.costs = costs
    self.serve_cost = serve_cost
    self.neighbors = neighbors
    self.load = 0
    self.idx = idx


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
          served (int): number of packets the node chose to serve
    """
    underflow = False
    overflow = False

    # a new packet is spawned
    spawned = np.random.binomial(n=1, p=self.spawn_rate)


    # if the load was 1 and the node both served and transmitted, serving
    # will not take place
    if (self.load - departures - served) < 0:
      served = 0

      if (self.load -departures) < 0:
        departures = 0

    # transition
    self.load = self.load - departures - served + spawned + arrivals

    # detect underflow
    if self.load < self.cutoff:
      underflow = True
      self.load = self.cutoff

    # detect overflow
    if self.load > (self.capacity):
      overflow = True
      self.load = self.capacity + 1

    return underflow, overflow, served





