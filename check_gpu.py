"""Utility to verify PyTorch installation and GPU availability."""

import torch


def main() -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Device: {device_name}")
    else:
        print("CUDA not available, running on CPU")


if __name__ == "__main__":
    main()
