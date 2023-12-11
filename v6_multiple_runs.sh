#!/bin/bash

source C:/Users/Kazim/anaconda3/etc/profile.d/conda.sh
conda activate vignetting_circle_detection_env

python v6_global_optim_terminal.py -t 2000 -m 100 -l 1 -n 50 --scheduler 0 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 100 -l 1 -n 50 --scheduler 1 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 200 -l 1 -n 50 --scheduler 0 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 200 -l 1 -n 50 --scheduler 1 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 100 -l 2 -n 50 --scheduler 0 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 100 -l 2 -n 50 --scheduler 1 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 200 -l 2 -n 50 --scheduler 0 --lr_scale 2
python v6_global_optim_terminal.py -t 2000 -m 200 -l 2 -n 50 --scheduler 1 --lr_scale 2
