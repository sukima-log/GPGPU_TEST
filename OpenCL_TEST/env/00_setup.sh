#!/bin/bash

# Nvidia GPU
sudo apt update -y
sudo apt upgrade -y
sudo apt autoremove -y

# ドライバインストール
sudo apt install nvidia-driver-580-open nvidia-opencl-dev -y

sudo apt install clinfo -y
