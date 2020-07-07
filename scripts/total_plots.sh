#!/bin/bash

# ----- plot Q-learning -----
echo "plotting for Q-learning"

#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 1 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 0 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 1 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 0 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 1 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 0 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 1 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/Qlearning 0 0 minimaxQ worst
#python3 ../scripts/plot_adversarial.py refactor/Qlearning Qlearning


#python3 ../scripts/plot_robust.py samples/Qlearning minimaxQ worst

#python3 ../scripts/plot_robust.py samples/Qlearning RomQ worst

#python3 ../scripts/plot_adversarial.py minimaxQ worst

#python3 ../scripts/plot_intermediate.py samples/Qlearning Qlearning

#python3 ../scripts/plot_intermediate.py samples/Qlearning minimaxQ

#python3 ../scripts/plot_intermediate.py samples/Qlearning RomQ


#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 1 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 0 Qlearning worst
#python3 ../scripts/plot_adversarial.py refactor/minimaxQ minimaxQ
#python3 ../scripts/plot_debug.py refactor/minimaxQ minimaxQ

#python3 ../scripts/plot_robust.py samples/minimaxQ Qlearning worst

#python3 ../scripts/plot_robust.py samples/minimaxQ minimaxQ worst

#python3 ../scripts/plot_robust.py samples/minimaxQ RomQ worst

#python3 ../scripts/plot_adversarial.py minimaxQ worst

#python3 ../scripts/plot_intermediate.py samples/minimaxQ Qlearning

#python3 ../scripts/plot_intermediate.py samples/minimaxQ minimaxQ

#python3 ../scripts/plot_intermediate.py samples/minimaxQ RomQ


# ----- plot RomQ -----
echo "plotting for Rom-Q"
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 1 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 0 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 1 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 0 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 1 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 0 0 minimaxQ worst
#python3 ../scripts/plot_adversarial.py refactor/RomQ RomQ
#python3 ../scripts/plot_heatmaps.py refactor/RomQ 1 1 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/RomQ  1 0 RomQ worst
#python3 ../scripts/plot_adversarial.py refactor/RomQ RomQ

#python3 ../scripts/plot_robust.py samples_final/RomQ Qlearning worst RomQ

#python3 ../scripts/plot_robust.py samples_final/RomQ minimaxQ worst RomQ

#python3 ../scripts/plot_robust.py samples_final/RomQ RomQ worst RomQ

#python3 ../scripts/plot_adversarial.py RomQ worst

#python3 ../scripts/plot_intermediate.py samples/RomQ Qlearning

#python3 ../scripts/plot_intermediate.py samples/RomQ minimaxQ

#python3 ../scripts/plot_intermediate.py samples/RomQ RomQ

# ----- plot minimaxQ -----
echo "plotting for minimaxQ"

#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 1 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 0 minimaxQ worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 1 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 0 RomQ worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 1 Qlearning worst
#python3 ../scripts/plot_heatmaps.py refactor/minimaxQ 0 0 Qlearning worst

# ----- group plots -----
echo "plotting group plots"
python3 ../scripts/plot_convergence.py refactor worst
python3 ../scripts/play_tournament.py
python3 ../scripts/plot_tournament.py
#python3 ../scripts/compare_perf.py
#python3 ../scripts/plot_convergence.py
#python3 ../scripts/compare_tournament.py

