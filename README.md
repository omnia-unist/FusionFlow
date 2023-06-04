# Table of Contents
- [Table of Contents](#table-of-contents)
- [Implementation of Accelerating Dataloader in Multi GPU project](#implementation-of-accelerating-dataloader-in-multi-gpu-project)
  - [Run Experiment](#run-experiment)
  - [Debug and Detailed Performance Checker](#debug-and-detailed-performance-checker)

# Implementation of Accelerating Dataloader in Multi GPU project

## Requirement
### Allinone
```console
CUDA_VERSION=1X.X
//Check out https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
DALI_CUDA_VERSION=1XX
conda install jupyter
sudo apt install libnuma-dev
pip install numa posix_ipc psutil matplotlib jupyterlab imgaug tensorflow-cpu==2.7.0 tensorflow_addons efficientnet_pytorch vit-pytorch sklearn git+https://github.com/wbaek/theconf@de32022f8c0651a043dc812d17194cdfd62066e8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=${CUDA_VERSION} -c pytorch
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda${DALI_CUDA_VERSION}
conda install -c conda-forge cupy cudatoolkit=${CUDA_VERSION}
```


### Torch
```console
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
```

### Tensorflow for tf.data
```console
pip install tensorflow-cpu==2.7.0 tensorflow_addons
pip install imgaug
```

### DALI
```console
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda1xx
conda install -c conda-forge cupy cudatoolkit=1x.x
```


## Before Running
1. Prevent File Open Error
```
ulimit -n unlimited
```

## Run Experiment
1. ./finegrained_DSAnalyzer.sh
   - Execute Main Experiment

```console
$ ./finegrained_DSAnalyzer.sh
```



## Debug and Detailed Performance Checker
1. ./fulltrace_main.sh
```console
$ ./fulltrace_main.sh
```