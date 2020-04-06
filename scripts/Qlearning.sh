#!/bin/bash


# ----- training -----
#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 10 --exec_attack_prob 0 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D

# ----- copy learned policies for tournament -----
#cp -r ../projects/samples_final/Qlearning2/policies/adversary_Qlearning ../projects/samples_final/minimaxQ/policies/adversary_Qlearning
#cp -r ../projects/samples_final/Qlearning2/policies/adversary_Qlearning ../projects/samples_final/RomQ/policies/adversary_Qlearning



# ----- evaluate optimal policies ----
python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 8  --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary RomQ

#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 10  --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary Qlearning

python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 7 --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary minimaxQ


# ----- evalute intermediate policies -----
python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 8  --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary RomQ


#python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 10  --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary Qlearning

python3 experiment_network.py --episodes 10000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project samples_final/Qlearning2 --trials 7  --exec_attack_prob 1 --method Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary minimaxQ
