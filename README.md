# Rom-Q

This repo contains my simulations for developing Rom-Q, a robust temporal difference learning algorithm for multi-agent systems. This project was developed during my research visit at the Intelligent and Autonomous Systems Group in CWI, under the supervision of Dr. Bloembergen and Dr. Kaisers. You can read more about the algorithm and simulations by reading the [report](report.pdf).

The project was developed using Python 3.7.6.

## How to use

To install all required libraries create a conda environment using:

`conda env create -f environment.yml`

File [experiment_network.py](source/experiment_network.py) is the main interface of the project. For example, the following call will simulate a simple network of two nodes having a capacity of 3 packets, where each node is an agent employing RomQ-learning for 100000 time steps in order to learn a robust policy. Results will be averaged over 10 independent trials and plots will be produced under folder "projects/myproject/plots".

`python3 experiment_network.py --project myproject --trials 10 --method RomQ --adversary RomQ --topology pair --capacity 3 --train_samples 1000000`

The directory [scripts](scripts) contains bash scripts for running all simulations and producing all plots used in the report that accompanied this project.
