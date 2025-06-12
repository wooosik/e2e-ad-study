"""Study Plan Code for E2E Autonomous Driving Study.

Each function below roughly corresponds to a weekly deliverable in the
study plan. The implementations are intentionally minimal and meant to
serve as starting points for further exploration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


# Week 1 ----------------------------------------------------------------------

def train_mnist_mlp(epochs: int = 1) -> float:
    """Train a simple 2-layer MLP on MNIST and return final accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Final MNIST accuracy: {accuracy:.4f}")
    return accuracy


# Week 3 ----------------------------------------------------------------------

def frame_to_tensor(image_path: str) -> torch.Tensor:
    """Load an RGB image with PIL and convert it to a normalized tensor."""
    from PIL import Image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    print(f"Tensor shape: {tensor.shape}")
    return tensor


# Week 4 ----------------------------------------------------------------------

def kinematic_bicycle_step(x: float, y: float, yaw: float, velocity: float, steering_angle: float, dt: float) -> tuple[float, float, float]:
    """Propagate a simple kinematic bicycle model one time step."""
    wheelbase = 2.5  # meters
    x += velocity * torch.cos(torch.tensor(yaw)) * dt
    y += velocity * torch.sin(torch.tensor(yaw)) * dt
    yaw += velocity / wheelbase * torch.tan(torch.tensor(steering_angle)) * dt
    return float(x), float(y), float(yaw)


def main() -> None:
    # Example usage of week 1 code
    if torch.cuda.is_available():
        print("CUDA device found. Training on GPU.")
    else:
        print("Training on CPU.")

    acc = train_mnist_mlp(epochs=1)
    print(f"Accuracy after 1 epoch: {acc:.4f}")

    # Week 3 example: convert an image (path required)
    sample_image = os.environ.get("SAMPLE_IMAGE")
    if sample_image and Path(sample_image).exists():
        frame_to_tensor(sample_image)
    else:
        print("Set SAMPLE_IMAGE env var to view image tensor conversion.")

    # Week 4 example: simple bicycle model step
    x, y, yaw = kinematic_bicycle_step(0.0, 0.0, 0.0, velocity=5.0, steering_angle=0.1, dt=0.1)
    print(f"Bicycle state after step: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")


if __name__ == "__main__":
    main()
