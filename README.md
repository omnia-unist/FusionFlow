# Table of Contents
- [Table of Contents](#table-of-contents)
- [Implementation of Accelerating Dataloader in Multi GPU project](#implementation-of-accelerating-dataloader-in-multi-gpu-project)
  - [Run Experiment](#run-experiment)
  - [Debug and Detailed Performance Checker](#debug-and-detailed-performance-checker)

# Implementation of Accelerating Dataloader in Multi GPU project

## Requirement
### Allinone
```console
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110==1.24.0
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install vit-pytorch==0.26.7
pip install git+https://github.com/wbaek/theconf.git
```



## Run Experiment
1. ./train.sh
   - Execute Main Experiment

```console
$ ./train.sh
```


