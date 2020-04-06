# ----- test  Qlearning -----

# against Q-learning
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &

# against minimaxQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &

# against RoMQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only 


# ----- test  RomQ -----

# against Q-learning
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ  --trials 5 --exec_attack_prob 1 --method RomQ  --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &


# against minimaxQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ  --trials 5 --exec_attack_prob 1 --method RomQ  --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &

# against RoMQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ  --trials 5 --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only


# ----- test  minimaxQ -----
# execute only 

# against Q-learning
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ  --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &

# against minimaxQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ  --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only &

# against RoMQ
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ  --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only



# ----- test  minimaxQ -----
# intemerdiate

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &


python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/minimaxQ  --trials 5 --exec_attack_prob 1 --method minimaxQ  --adversary RomQ --test_episodes 1000 --topology pair  --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate 


# ----- test RomQ -----
# intermediate
python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ --trials 5 --exec_attack_prob 1 --method RomQ  --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ --trials 5 --exec_attack_prob 1 --method RomQ  --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/RomQ  --trials 5 --exec_attack_prob 1 --method RomQ  --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate

# ----- test Qlearning -----
#intermediate

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary Qlearning --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary minimaxQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate &

python3 experiment_network.py --episodes 20000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project new/Qlearning --trials 5 --exec_attack_prob 1 --method Qlearning --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate



