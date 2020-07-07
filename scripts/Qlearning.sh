#!/bin/bash


# ----- training -----
#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary Qlearning --topology pair --capacity 3 --network_type D --train_samples 1000000

# ----- copy learned policies for tournament -----
#cp -r ../projects/refactor/Qlearning/policies/adversary_Qlearning ../projects/refactor/minimaxQ/policies/adversary_Qlearning
#cp -r ../projects/refactor/Qlearning/policies/adversary_Qlearning ../projects/refactor/RomQ/policies/adversary_Qlearning

# ----- PROB VS DETERM -----

# ----- evaluate optimal policies ----
#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --eval_attack_prob 1

#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary minimaxQ --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --eval_attack_prob 1

#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary RomQ --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --eval_attack_prob 1


# ----- evalute intermediate policies -----
python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1

python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary minimaxQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --evaluate_interm

python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary RomQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --evaluate_interm


# ----- DETERM VS DETERM -----
# ----- evalute intermediate policies -----
#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_adv

#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary minimaxQ_determ --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --eval_attack_prob 1 --evaluate_interm --determ_adv

#python3 experiment_network.py --project refactor/Qlearning --trials 10 --method Qlearning --adversary RomQ_determ --topology pair --capacity 3 --network_type D --eval_samples 30000 --evaluate --eval_attack_prob 1 --evaluate_interm --determ_adv
