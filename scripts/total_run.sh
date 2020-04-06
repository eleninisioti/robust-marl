#!/bin/bash
# ----- test Q-learning -----

# against Q-learning
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against minimaxQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against RoMQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/Qlearning --trials 10 --exec_attack_prob 1 --method Qlearning --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate


# ----- test minimaxQ -----

# against Q-learning
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against minimaxQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against RoMQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/minimaxQ --trials 10 --exec_attack_prob 1 --method minimaxQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate


# ----- test RoMQ -----

# against Q-learning
python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only


python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against minimaxQ
python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# against RoMQ
python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only

python3 experiment_network.py --episodes 40000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project project/RomQ --trials 10 --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate
