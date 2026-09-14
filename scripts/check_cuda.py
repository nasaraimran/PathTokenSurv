from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that this Python environment can execute PyTorch CUDA kernels.")
    parser.add_argument("--require", action="store_true", help="Exit non-zero when CUDA is unavailable or a smoke kernel fails.")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("Device: CPU")
        if args.require:
            raise SystemExit(2)
        return

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {major}.{minor}")
    print(f"GPU memory: {props.total_memory / (1024**3):.2f} GiB")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"Compiled CUDA architectures: {', '.join(arch_list)}")

    if arch_list and arch not in arch_list:
        print(f"WARNING: this wheel does not list native support for {arch}.")
        if args.require:
            raise SystemExit(3)

    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = a @ b
        torch.cuda.synchronize()
        if not torch.isfinite(c).all().item():
            raise RuntimeError("CUDA smoke result contains non-finite values.")
        print("CUDA kernel smoke test: PASS")
        print(f"Allocated after smoke test: {torch.cuda.memory_allocated(device) / (1024**2):.1f} MiB")
    except Exception as exc:
        print(f"CUDA kernel smoke test: FAIL ({exc})")
        if args.require:
            raise SystemExit(4) from exc


if __name__ == "__main__":
    main()
