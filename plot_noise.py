from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import make_loader
from schedule import build_schedule, q_sample

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(exist_ok=True)


def to_display(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]，供 imshow 显示。"""
    return ((x + 1) / 2).clamp(0, 1)


def main():
    torch.manual_seed(0)
    sched = build_schedule()

    loader = make_loader(batch_size=8)
    x0, labels = next(iter(loader))
    x0 = x0[:5]
    labels = labels[:5]

    rows = []  # (t, 该 t 下的 5 张图)
    for t in [0, 100, 300, 600, 999]:
        if t == 0:
            xt = x0.clone()  # t=0 还没加噪，直接显示原图
        else:
            xt = q_sample(x0, t, sched.alpha_bar)
        rows.append((t, xt))

    # 理论噪声强度 sqrt(1-alpha_bar)，与图像互证
    for t, _ in rows:
        if t > 0:
            ab = sched.alpha_bar[t - 1].item()
            print(f"t={t:>4}  noise std = {((1 - ab) ** 0.5):.4f}")

    # ---- 画图 ----
    n_rows, n_cols = len(rows), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))

    for r, (t, xt) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            img = to_display(xt[c]).squeeze(0)  # (1, 28, 28) -> (28, 28)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"label={labels[c].item()}")
        axes[r][0].set_ylabel(f"t={t}", fontsize=11)

    fig.suptitle("Forward diffusion on MNIST", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = IMG_DIR / "noise_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
