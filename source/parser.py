""" This file contains all input flags used by experiment_network.
"""
# ----- generic imports -----
import argparse


def parse_flags():
  parser = argparse.ArgumentParser()

  # ----- related to modelling the game -----
  parser.add_argument('--N',
                      help='Number of nodes',
                      type=int,
                      default=100)

  parser.add_argument('--clow',
                      help='Punishment for underflow',
                      type=int,
                      default=0)

  parser.add_argument('--chigh',
                      help='Punishment for overflow',
                      type=int,
                      default=100)

  parser.add_argument('--utility',
                      help='Reward for being alive (not over-flown or '
                           'under-flown)',
                      type=int,
                      default=8)

  parser.add_argument('--capacity',
                      help='Capacity of nodes',
                      type=int,
                      default=3)

  parser.add_argument('--K',
                      help='Number of adversaries',
                      type=int,
                      default=1)

  parser.add_argument('--topology',
                      help='The network topology. Choose between Ring and '
                           'star.',
                      type=str,
                      default="ring")

  parser.add_argument('--network_type',
                      help='The type of network determines the modelling of '
                           'nodes. Choose between A, B and C.',
                      type=str,
                      default="A")

  # ----- related to learning parameters ------
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

  parser.add_argument('--algorithm',
                      help='Indicates the learning algorithm used. Choose '
                           'between Qlearning, minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--adversary',
                      help='Choose adversarial policy. Choices are Qlearning '
                           ' minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--horizon',
                      help='Number of iterations in episode',
                      type=int,
                      default=50)

  # ----- configuring the simulation ------
  parser.add_argument('--project',
                      help='Name of project',
                      type=str,
                      default="temp")

  parser.add_argument('--trials',
                      help='Number of monte carlo trials.',
                      type=int,
                      default=5)

  parser.add_argument('--epochs',
                      help='Number of epochs (for saving intermediate '
                           'results).',
                      type=int,
                      default=5)

  parser.add_argument('--train_samples',
                      help='Number of training samples',
                      type=int,
                      default=1000000)

  parser.add_argument('--eval_samples',
                      help='Number of evaluation samples',
                      type=int,
                      default=20000)

  parser.add_argument('--evaluate',
                      help='Evaluate existing policies.',
                      default=False,
                      action="store_true")

  parser.add_argument('--train',
                      help='Train new policies.',
                      default=False,
                      action="store_true")

  parser.add_argument('--evaluate_interm',
                      help='Indicates whether all intermediate trained '
                           'policies will be evaluated. Otherwise, only the '
                           'policy after convergence is evaluated.',
                      default=False,
                      action="store_true")

  parser.add_argument('--adversarial_interm',
                      help='Indicates whether intermediate adversarial '
                           'policies will be computed, stored and used for '
                           'evaluation.',
                      default=False,
                      action="store_true")

  parser.add_argument('--attack_type',
                      help='Choose between rand, rand_nodes, rand_actions and'
                           ' worst.',
                      type=str,
                      default="worst")

  parser.add_argument('--eval_attack_prob',
                      help='Probability of attack during evaluation.',
                      type=float,
                      default=1)

  args = parser.parse_args()

  return args
