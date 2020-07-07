""" This file contains all input flags used by experiment_network.
"""
# ----- generic imports -----
import argparse


def parse_flags():
  parser = argparse.ArgumentParser()

  parser.add_argument('--N',
                      help='Number of nodes',
                      type=int,
                      default=100)

  parser.add_argument('--clow',
                      help='Punishment for underflow',
                      type=int,
                      default=0)

  parser.add_argument('--evaluate',
                      help='Evaluate a learned policy.',
                      default=False,
                      action="store_true")

  parser.add_argument('--determ_execution',
                      help='Deterministic execution of policies.',
                      default=False,
                      action="store_true")

  parser.add_argument('--determ_adv',
                      help='Deterministic adversarial policies.',
                      default=False,
                      action="store_true")

  parser.add_argument('--explore_attack',
                      help='Explore sub-optimal attacks (only for Rom-Q)',
                      default=0,
                      type=float)

  parser.add_argument('--evaluate_interm',
                      help='Indicates whether evaluation should be done all '
                           'intermediate trained agents. Otherwise, only the '
                           'final policy is evaluated.',
                      default=False,
                      action="store_true")

  parser.add_argument('--adversarial_interm',
                      help='Indicates whether intermediate policies will be '
                           'computed, stored and used for evaluation.',
                      default=False,
                      action="store_true")

  parser.add_argument('--chigh',
                      help='Punishment for overflow',
                      type=int,
                      default=100)

  parser.add_argument('--utility',
                      help='Reward for being alive',
                      type=int,
                      default=8)

  parser.add_argument('--attack_type',
                      help='Choose betweetn randa, randb and worst.',
                      type=str,
                      default="worst")

  parser.add_argument('--K',
                      help='Number of adversaries',
                      type=int,
                      default=1)

  parser.add_argument('--eval_attack_prob',
                      help='Probability of attack during execution.',
                      type=float,
                      default=0)

  parser.add_argument('--learning_rate',
                      help='Learning rate for temporal difference learning.',
                      type=float,
                      default=0.01)

  parser.add_argument('--discount_factor',
                      help='Discount factor for temporal difference learning.',
                      type=float,
                      default=0.9)

  parser.add_argument('--epsilon',
                      help='Exploration rate for temporal difference learning.',
                      type=float,
                      default=0.1)

  parser.add_argument('--capacity',
                      help='Capacity of nodes',
                      type=int,
                      default=3)

  parser.add_argument('--trials',
                      help='Number of Monte Carlo trials.',
                      type=int,
                      default=5)

  parser.add_argument('--method',
                      help='Indicates the learning method used. Choose '
                           'between Qlearning, minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--adversary',
                      help='Choose adversarial policy. Choices are Qlearning '
                           ' minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--train_samples',
                      help='Number of training samples',
                      type=int,
                      default=50000)

  parser.add_argument('--eval_samples',
                      help='Number of evaluation samples',
                      type=int,
                      default=50000)

  parser.add_argument('--project',
                      help='Name of project',
                      type=str,
                      default="temp")

  parser.add_argument('--explore',
                      help='Exploration technique to use.',
                      type=str,
                      default="egreedy")

  parser.add_argument('--topology',
                      help='The network topology. Choose between Ring and '
                           'star.',
                      type=str,
                      default="ring")

  parser.add_argument('--network_type',
                      help='Type of network. Choose between A, B and C.',
                      type=str,
                      default="A")

  parser.add_argument('--seed',
                      help='Seed used for generating random numbers.',
                      type=int,
                      default=0)

  parser.add_argument('--horizon',
                      help='Number of iterations in episode',
                      type=int,
                      default=50)

  args = parser.parse_args()

  return args
