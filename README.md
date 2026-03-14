
# CUDA and OpenCL GEMM Test

# 環境
- OS : Ubuntu 24.04.4 LTS
- GPU : NVIDIA GeForce RTX 2060 Super
- CUDA : 13.0
- Compiler : nvcc, g++


# GPU認識確認

`$lspci | grep -i nvidia`

出力例
```
01:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2060 SUPER] (rev a1)
01:00.1 Audio device: NVIDIA Corporation TU106 High Definition Audio Controller (rev a1)
01:00.2 USB controller: NVIDIA Corporation TU106 USB 3.1 Host Controller (rev a1)
01:00.3 Serial bus controller: NVIDIA Corporation TU106 USB Type-C UCSI Controller (rev a1)
```

# ドライバ確認

`ubuntu-drivers devices`

```
== /sys/devices/pci0000:00/xxxxxxx/xxxxxxxxxxxx ==
modalias : pci:xxxxxxxxxxxxxxxxxxxxxxxxxxx
vendor   : NVIDIA Corporation
model    : TU106 [GeForce RTX 2060 SUPER]
driver   : nvidia-driver-535-server-open - distro non-free
driver   : nvidia-driver-580-open - distro non-free recommended
driver   : nvidia-driver-535-open - distro non-free
driver   : nvidia-driver-590-open - distro non-free
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-535 - distro non-free
driver   : nvidia-driver-570-server-open - distro non-free
driver   : nvidia-driver-580-server-open - distro non-free
driver   : nvidia-driver-570-open - distro non-free
driver   : nvidia-driver-570 - distro non-free
driver   : nvidia-driver-570-server - distro non-free
driver   : nvidia-driver-590 - distro non-free
driver   : nvidia-driver-590-server - distro non-free
driver   : nvidia-driver-580 - distro non-free
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-580-server - distro non-free
driver   : nvidia-driver-590-server-open - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```


# GPU動作確認

`nvidia-smi`
```
Sat Mar 14 18:31:01 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2060 ...    Off |   00000000:01:00.0 Off |                  N/A |
|  0%   44C    P8             23W /  175W |       1MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

# CUDA Toolkit 確認
`nvcc --version`

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

# GPU動作確認

`watch -n 0.5 nvidia-smi`


# OpenCL プラットフォーム確認 GPU OpenCL デバイス確認

`clinfo`

`clinfo | grep "Platform\|Device"`

# OpenCL ヘッダとライブラリの確認

`dpkg -l | grep opencl`


# RTX2060 Spec

| 項目	    | 数値      |
|   --      |   --      |
| CUDAコア  |	1920    |
| VRAM	    | 6GB       |
| Compute Capability |	7.5 |

# GPU メモリ階層

```
DRAM (Global Memory) (400~800 cycle)
        ↓
L2 Cache (200 cycle)
        ↓
SM (Streaming Multiprocessor)
        ↓
Shared Memory (20 cycle)
        ↓
Registers (1 cycle)
        ↓
CUDA Core
```