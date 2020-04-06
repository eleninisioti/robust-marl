python3 experiment_network.py --episodes 100000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project debug/RomQ_long_D --trials 5  --exec_attack_prob 1 --method RomQ --adversary RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D

python3 experiment_network.py --episodes 100000 --horizon 50 --epsfilon 0.1 --learning_rate 0.01 --project debug/RomQ_long --trials 5  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --adversary RomQ 

python3 experiment_network.py --episodes 100000 --horizon 50 --epsilon 0.1 --learning_rate 0.01 --project debug/RomQ_long --trials 5  --exec_attack_prob 1 --method RomQ --test_episodes 1000 --topology pair --K 1 --N 2 --capacity 3 --network_type D --execute_only --test_intermediate --adversary RomQ


python3 ../scripts/plot_robust.py debug/RomQ_long RomQ worst
python3 ../scripts/plot_intermediate.py debug/RomQ_long RomQ worst
