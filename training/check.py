import subprocess

import torch


print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

try:
    result = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
    print("\nnvidia-smi output:")
    print(result[:500])
except Exception as exc:
    print("nvidia-smi failed. NVIDIA driver or CUDA may be unavailable:", exc)

if torch.cuda.is_available():
    print("\nGPU model:", torch.cuda.get_device_name(0))
else:
    print("\nNo CUDA device available")
