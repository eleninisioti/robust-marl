#!/bin/bash

# ----- train RoMQ -----
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project final/RomQ --trials 5 --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D

# ----- train Qlearning -----
#python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 0 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D




# ----- train minimaxQ -----
#python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ --trials 5 --exec_attack_prob 0 --method minimaxQ --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D


