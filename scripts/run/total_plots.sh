#!/bin/bash

# Note: Running this script requires that all necessary data are available under projects/simuls. It needs to be run within the source directory of the repo.

# ----- plot Q-learning -----
echo "plotting for Q-learning"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plots/plot_heatmaps.py simuls/Qlearning 0 1 Qlearning worst
python3 ../scripts/plots/plot_heatmaps.py simuls/Qlearning 0 1 RomQ worst
python3 ../scripts/plots/plot_heatmaps.py simuls/Qlearning 0 1 minimaxQ worst

# plot adversarial policies
python3 ../scripts/plots/plot_adversarial.py simuls/Qlearning worst

# ----- plot minimax-Q -----
echo "plotting for minimax-Q"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plots/plot_heatmaps.py simuls/minimaxQ 0 1 Qlearning worst
python3 ../scripts/plots/plot_heatmaps.py simuls/minimaxQ 0 1 RomQ worst
python3 ../scripts/plots/plot_heatmaps.py simuls/minimaxQ 0 1 minimaxQ worst

python3 ../scripts/plots/plot_adversarial.py simuls/minimaxQ worst

# ----- plot Q-learning -----
echo "plotting for Rom-Q"

# plot heatmaps of performance during evaluation for all training epochs 
python3 ../scripts/plots/plot_heatmaps.py simuls/RomQ 0 1 Qlearning worst
python3 ../scripts/plots/plot_heatmaps.py simuls/RomQ 0 1 RomQ worst
python3 ../scripts/plots/plot_heatmaps.py simuls/RomQ 0 1 minimaxQ worst

python3 ../scripts/plots/plot_adversarial.py simuls/RomQ worst

# ----- group plots -----
echo "plotting group plots"
#python3 ../scripts/plot_convergence.py simuls worst
python3 ../scripts/plots/play_tournament.py simuls
python3 ../scripts/plots/plot_tournament.py simuls


