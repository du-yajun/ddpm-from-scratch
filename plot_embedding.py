from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from embedding import time_embedding

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)

TS = [1, 10, 100, 300, 600, 999]


def main():
    t = torch.tensor(TS)
    emb = time_embedding(t, dim=128)  # (6, 128)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    im = ax.imshow(emb.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(len(TS)))
    ax.set_yticklabels([f"t={v}" for v in TS])
    ax.set_xlabel("embedding dimension")
    ax.set_title("Sinusoidal time embedding")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out = IMG_DIR / "time_embedding_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
