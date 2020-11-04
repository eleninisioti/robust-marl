#!/bin/bash

# Note: Running this script requires that all necessary data are available under projects/simuls

# ----- plot Q-learning -----
echo "plotting for Q-learning"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plot_heatmaps.py simuls/Qlearning 0 1 Qlearning worst
python3 ../scripts/plot_heatmaps.py simuls/Qlearning 0 1 RomQ worst
python3 ../scripts/plot_heatmaps.py simuls/Qlearning 0 1 minimaxQ worst

# plot adversarial policies
python3 ../scripts/plot_adversarial.py Qlearning worst

# ----- plot minimax-Q -----
echo "plotting for minimax-Q"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plot_heatmaps.py simuls/minimaxQ 0 1 Qlearning worst
python3 ../scripts/plot_heatmaps.py simuls/minimaxQ 0 1 RomQ worst
python3 ../scripts/plot_heatmaps.py simuls/minimaxQ 0 1 minimaxQ worst

python3 ../scripts/plot_adversarial.py minimaxQ worst

# ----- plot Q-learning -----
echo "plotting for Rom-Q"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plot_heatmaps.py simuls/RomQ 0 1 Qlearning worst
python3 ../scripts/plot_heatmaps.py simuls/RomQ 0 1 RomQ worst
python3 ../scripts/plot_heatmaps.py simuls/RomQ 0 1 minimaxQ worst

python3 ../scripts/plot_adversarial.py RomQ worst

# ----- group plots -----
echo "plotting group plots"
python3 ../scripts/plot_convergence.py simuls worst
python3 ../scripts/play_tournament.py simuls
python3 ../scripts/plot_tournament.py simuls


