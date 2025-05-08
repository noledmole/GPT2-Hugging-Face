#!/bin/bash

# System dependencies
sudo apt update
sudo apt install -y git unzip htop python3-pip

# REMOVE keras, tensorflow system installs to avoid conflicts
sudo apt purge -y python3-keras python3-tensorflow-cuda lambda-stack-cuda
sudo apt autoremove -y

# Upgrade pip and install from PyPI
pip3 install --upgrade pip

# Install only PyTorch-compatible versions of Transformers
pip3 install \
  torch \
  transformers[torch]==4.51.3 \
  datasets==3.6.0 \
  accelerate==1.6.0 \
  wandb==0.19.11 \
  tokenizers==0.21.1 \
  safetensors>=0.4.3 \
  tqdm>=4.27
