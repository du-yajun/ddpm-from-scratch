import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=tf
)

data_loader = DataLoader(train_set, batch_size=256, shuffle=True)

x_max = float("-inf")
x_min = float("inf")
for x, _ in data_loader:
    x_max = max(x_max, x.max().item())
    x_min = min(x_min, x.min().item())

print(f"full dataset range = [{x_min:.4f}, {x_max:.4f}]")
