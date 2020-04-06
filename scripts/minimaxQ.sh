#!/bin/bash

# ----- training -----
#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 10  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --adversary minimaxQ


# ----- copy learned policies for tournament -----
#cp -r ../projects/samples_final/minimaxQ/policies/adversary_minimaxQ ../projects/samples_final/RomQ/policies/adversary_minimaxQ
#cp -r ../projects/samples_final/minimaxQ/policies/adversary_minimaxQ ../projects/samples_final/Qlearning/policies/adversary_minimaxQ

# ----- evaluate optimal policies ----
#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary minimaxQ


#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary Qlearning

#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary RomQ


# ----- evaluate intermediate policies ----
python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary minimaxQ


python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary Qlearning

python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/minimaxQ --trials 7  --exec_attack_prob 1 --method minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary RomQ


