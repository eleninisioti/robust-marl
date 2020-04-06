#!/bin/bash
# ----- train RoMQ -----
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project fix/RomQ --trials 5 --exec_attack_prob 0 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D



