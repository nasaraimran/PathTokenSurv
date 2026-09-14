# Windows GPU Setup for PathTokenSurv v1.5.6

This guide is intended for the reported NVIDIA GeForce GTX 1070 Ti system.

## Why CUDA 12.6

The GTX 1070 Ti is a Pascal-generation GPU. CUDA 13 removed toolkit/library support for Pascal, so use a CUDA 12.x PyTorch wheel. PyTorch publishes Windows CUDA 12.6 wheels. For this Pascal GPU, the conservative primary environment is torch 2.12.1 + cu126, because PyTorch explicitly documents CUDA 12.6 as the supported wheel for older Pascal/Volta architectures in the 2.12 release.

The `CUDA Version` printed by `nvidia-smi` is a driver compatibility value. It does not mean the Python environment has a CUDA-enabled PyTorch build. A version string such as `torch 2.13.0+cpu` means PyTorch itself is CPU-only.

## Installation in the existing virtual environment

Activate the project virtual environment, then run:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-build-isolation
```

`torchvision` and `torchaudio` are not required by PathTokenSurv.

## Verify the actual CUDA runtime

```powershell
python .\scripts\check_cuda.py --require
```

A successful GTX 1070 Ti setup should report CUDA available, an NVIDIA GPU, compute capability 6.1, a CUDA 12.6 PyTorch runtime, and `CUDA kernel smoke test: PASS`.

You can also run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')"
```

## Start the fresh fold-1 manuscript run

Use a new output directory and require CUDA explicitly:

```powershell
python .\scripts\run_outer_fold.py `
    --config .\configs\tcga_pancancer_raw.json `
    --outer-folds 5 `
    --fold 1 `
    --validation-fraction 0.15 `
    --output-dir .\outputs\tcga_outer_fold1_v1.5.6_cuda `
    --require-cuda
```

Do not use the old v1.5.5 checkpoint for the primary manuscript run. Keep it as a diagnostic/safety artifact.

## Recovery after an interruption

v1.5.6 writes `last_model.pt` after every completed epoch. Resume the same v1.5.6 run with:

```powershell
python .\scripts\run_outer_fold.py `
    --config .\configs\tcga_pancancer_raw.json `
    --outer-folds 5 `
    --fold 1 `
    --validation-fraction 0.15 `
    --output-dir .\outputs\tcga_outer_fold1_v1.5.6_cuda `
    --resume-from .\outputs\tcga_outer_fold1_v1.5.6_cuda\last_model.pt `
    --require-cuda
```
