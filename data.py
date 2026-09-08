# MNIST 数据加载

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 基于本文件的位置定位 data 目录，而不是依赖"当前运行目录"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_mnist(train: bool = True):
    """download=True 时，若本地已有数据文件则不会真的联网"""
    return datasets.MNIST(
        root=str(DATA_ROOT),
        train=train,
        download=True,
        transform=get_transform(),
    )


def make_loader(batch_size: int = 256, train: bool = True, shuffle: bool = True):
    dataset = load_mnist(train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    loader = make_loader(batch_size=256)

    x_max = float("-inf")
    x_min = float("inf")
    for x, _ in loader:
        x_max = max(x_max, x.max().item())
        x_min = min(x_min, x.min().item())

    print(f"full dataset range = [{x_min:.4f}, {x_max:.4f}]")
