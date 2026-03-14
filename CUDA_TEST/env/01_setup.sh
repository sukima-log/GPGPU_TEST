#!/bin/bash

# GPU動作確認
nvidia-smi

# ツールキット インストール
sudo apt install nvidia-cuda-toolkit

# 確認
nvcc --version