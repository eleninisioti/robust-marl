#!/bin/bash

# ----- training -----
#python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ --topology pair --capacity 3 --network_type D --train_samples 1000000 --determ_execution --determ_adv

# ----- DETERM VS DETERM -----

# ----- evaluate optimal policies ----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_execution --determ_adv

#python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_execution --determ_adv

#python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_execution --determ_adv


# ----- evalute intermediate policies -----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_execution --determ_adv

#python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_execution --determ_adv

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_execution --determ_adv


# ----- PROB VS PROB -----
echo "Beginning prob vs prob"
# ----- evaluate optimal policies ----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 


# ----- evalute intermediate policies -----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 


# ----- DETERM VS PROB -----

# ----- evaluate optimal policies ----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_adv

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_adv

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --eval_attack_prob 1 --determ_adv


# ----- evalute intermediate policies -----
python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary minimaxQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_adv

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_adv

python3 experiment_network.py --project refactor/minimaxQ --trials 10 --method minimaxQ --adversary RomQ_determ --topology pair --capacity 3 --network_type D --eval_samples 10000 --evaluate --evaluate_interm --eval_attack_prob 1 --determ_adv

