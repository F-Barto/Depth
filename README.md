<div align="center">   
 
# Depth  

[![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch Lightning](https://img.shields.io/badge/pytorch%20lightning-7.1.0-blueviolet.svg)]()

**Note: this is a repo for personnal use. I don't have a ssh connection I have to push pull on remote server at every 
little modification; hence the dirty commits. The doc is not complete and may even be erroneous. In addition, what works 
today may not works tomorrow. Lastly, the scripts undar ``tasks/`` are made for the INRIA's cluster.**

[x] Train self-sup Monodepth
[x] Train semi-sup Monodepth
[ ] Train self-sup Packnet
[ ] Train semi-sup Packnet
[ ] Train self-sup i-ResNet + ESPCN
[-] Train multi-cam self-sup Monodepth


</div>
 
## Description
The present repo uses pytorch-lightning and wandb to facilitate research exploration and experiment tracking.  
It is aimed for quick training and evaluation of self-supervised depth estimation methods. 


## Repo organisation

* data-preparation
* networks
* models
* train.py

# How to run   

## Installation
First, clone the repo
```bash
# clone project   
git clone https://github.com/F-Barto/Depth
cd Depth
```

Then, install dependencies and activate env.
```bash
# create conda env (or use your usual env management system) and install dependancies 
conda env create --name Depth_env -f environment.yml
conda activate Depth_env
 ```  

## Data-preparation

DL Kitti
DL Argo
Install argo api

## Running the code:

### GPU / Multi-GPU
````bash
./tasks/run_self_supervised_packnet.py # change the --gpus '0' to --gpus '0,1,2' if you want to use multiple gpus
````

If you want to use 16-bit (mixed precision) training, install the following environment
```bash
conda env create --name Depth_env_mixed -f environment_mixed.yml
conda activate Depth_env_mixed
```

Notes:
- We uses the nightly version of Pytorch for a native support of NVIDIA-apex throught torch.cuda.amp as is the 
recommended way to go moving forward for mixed-precision training. 
- torch.cuda.amp will be officially supported in Pytorch >=1.6 version.<br>
- During the env install, conflicts are expected and conda solving them takes some time. 
    Hence installation can take ~30 min.


The four optimization levels:
- O0 (FP32 training): basically a no-op. Everything is FP32 just as before.
- O1 (Conservative Mixed Precision): only some whitelist ops are done in FP16.
- O2 (Fast Mixed Precision): this is the standard mixed precision training.<br>
    It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights.
- O3 (FP16 training): full FP16. Passing keep_batchnorm_fp32=True can speed things up as cudnn batchnorm is faster anyway.

Note: Some non-Volta cards (like the P100) can benefit from half-precision arithmetic for certain networks,
 but the numerical stability is much less reliable (even with Apex tools)

## Generated output files/dirs

# List of Depth Estimation Methods Implemented
Please cite the methods below if you use them.

If you want to load weights from PackNet you have to also install yacs and then:
>>> import torch
>>> path = "./ResNet18_MR_selfsup_K.ckpt"
>>> ckpt = torch.load(path, map_location='cpu')
>>> list(ckpt.keys())
['config', 'state_dict']
>>> list(ckpt['state_dict'].keys())
['model.depth_net.encoder.encoder.conv1.weight', 'model.depth_net.encoder.encoder.bn1.weight', ...,
 'model.pose_net.decoder.net.3.weight', 'model.pose_net.decoder.net.3.bias']


oarsub -p "(gpumem > 12000) and (gpumodel!='k40m')" -l "walltime=0:30:0" "/home/clear/fbartocc/depth_project/Depth/tasks/run_self_supervised_monodepth.sh --fast_dev_run"