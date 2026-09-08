from dataclasses import dataclass

import torch


@dataclass
class Schedule:
    T: int
    # 数组下标 i 对应论文第 i + 1 步
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor


def build_schedule(
    T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Schedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return Schedule(T=T, betas=betas, alphas=alphas, alpha_bar=alpha_bar)


if __name__ == "__main__":
    s = build_schedule()
    print("beta_1 = ", s.betas[0].item())
    print("beta_1000 = ", s.betas[-1].item())
    print("alpha_bar_1 = ", s.alpha_bar[0].item())
    print("alpha_bar_1000 = ", s.alpha_bar[-1].item())
