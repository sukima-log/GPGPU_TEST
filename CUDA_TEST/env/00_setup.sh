#!/bin/bash


# ドライバインストール
ubuntu-drivers devices

sudo apt update -y
sudo apt upgrade -y
sudo apt autoremove -y

# 任意のドライバーへ変更
sudo apt install nvidia-driver-580-open

sudo reboot




