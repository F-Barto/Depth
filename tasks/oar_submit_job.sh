#! /bin/bash

# Inria cluster specifics
source ~/.bashrc
source gpu_setVisibleDevices.sh

conda activate Depth_env

#WANBKEY=$(<../wandb.key) # load the key from a file at root dir
#wandb login $WANBKEY

python -u ../train.py "$@"