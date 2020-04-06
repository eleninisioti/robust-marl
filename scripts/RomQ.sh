#!/bin/bash

# ----- training -----
#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D

#cp -r ../projects/samples_final/RomQ/policies/adversary_RomQ ../projects/samples_final/minimaxQ/policies/adversary_RomQ
#cp -r ../projects/samples_final/RomQ/policies/adversary_RomQ ../projects/samples_final/Qlearning/policies/adversary_RomQ


# ----- evaluate optimal policies ----
#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 8  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary RomQ

#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 8  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary Qlearning

#python3 experiment_network.py --episodes 2000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 7  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary minimaxQ


# ----- evaluate intermediate policies ----
python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 8  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary RomQ


python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 8  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary Qlearning

python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/RomQ --trials 7  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary minimaxQ
