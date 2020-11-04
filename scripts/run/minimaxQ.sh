# train optimal and adversarial policies
python3 experiment_network.py --project simuls/minimaxQ --trials 40 --algorithm minimaxQ --adversary minimaxQ --topology pair --capacity 3 --network_type A --train_samples 1000000 --train --eval_attack_prob 1 

# Note: You can remove flag --evaluate_interm to only evalute the policy after convergence (but you won't be able to plot convergence with training time)
# evaluate optimal policies against minimax-Q adversaries 
python3 experiment_network.py --project simuls/minimaxQ --trials 40 --algorithm minimaxQ --adversary minimaxQ --topology pair --capacity 3 --network_type A --train_samples 1000000 --evaluate --eval_attack_prob 1 --evaluate_interm

# Note: to run the commands below you need to make sure that the corresponding files are under policies/adversary_Qlearning and policies/adversary_RomQ

# evaluate optimal policies against Q-learning adversaries 
python3 experiment_network.py --project simuls/minimaxQ --trials 40 --algorithm minimaxQ --adversary Qlearning --topology pair --capacity 3 --network_type A --train_samples 1000000 --evaluate --eval_attack_prob 1 --evaluate_interm

# evaluate optimal policies against RoM-Q adversaries 
python3 experiment_network.py --project simuls/minimaxQ --trials 40 --algorithm minimaxQ --adversary RomQ --topology pair --capacity 3 --network_type A --train_samples 1000000 --evaluate --eval_attack_prob 1 --evaluate_interm


