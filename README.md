# FusionFlow 

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Run Experiment with FusionFlow](#run-experiment-with-fusionflow)
  - [FusionFlow Torch Installation](#fusionflow-torch-installation)
  - [Run Experiment](#run-experiment)
  
# Run Experiment with FusionFlow

## Requirement
### Allinone
```sh
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110==1.26.0
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install vit-pytorch==0.26.7
pip install git+https://github.com/wbaek/theconf.git
```


## FusionFlow Torch Installation 
[https://github.com/omnia-unist/FusionFlow_PyTorch]

```sh
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
(install vit-pytorch kind of source build)

pip unistall torch (clean only torch)

python setup.py develop

```

## FusionFlow DALI Installation
[https://github.com/omnia-unist/FusionFlow_DALI]
```sh

cd wheelhouse
pip install nvidia_dali_cuda110-1.26.0.dev0-12345-py3-none-manylinux2014_x86_64.shl

```

If you want to install from the source code

- When compliation, using CUDA 11.0, CUDA 11.8 and CUDA 12.1 (other 11.x is not supported)

[Question about installation on DALI with dockerfile · Issue #4814 · NVIDIA/DALI](https://github.com/NVIDIA/DALI/issues/4814#issuecomment-1525918590)

- Compilation at the docker directory

```
sudo CUDA_VERSION=11.8 PYVER=3.8 BUILD_TEST=0 ./build.sh
```

- Made .wheel file to install (uninstall is necessary when the package is already installed

```python
pip uninstall nvidia-dali-cuda110
```

```jsx
pip install ../wheelhouse/nvidia_dali_cuda110-1.26.0.dev0-12345-py3-none-manylinux2014_x86_64.whl
```



## Run Experiment
1. ./train.sh
   - Execute Main Experiment

```console
$ ./train.sh
```


