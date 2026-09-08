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


# q_sample
def q_sample(x_0, t, alpha_bar, noise=None):
    B = x_0.shape[0]
    if isinstance(t, int):
        t = torch.full((B,), t, dtype=torch.long)

    assert (t >= 1).all() and (t <= len(alpha_bar)).all(), (
        "t 必须在 [1, len(alpha_bar)] 之间"
    )

    if noise is None:
        noise = torch.randn_like(x_0)

    alpha_bar_t = alpha_bar[t - 1].to(x_0)
    view = (B,) + (1,) * (x_0.dim() - 1)
    x_t = (
        torch.sqrt(alpha_bar_t).view(view) * x_0
        + torch.sqrt(1 - alpha_bar_t).view(view) * noise
    )
    return x_t


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


if __name__ == "__main__":
    s = build_schedule()
    # print("beta_1 = ", s.betas[0].item())
    # print("beta_1000 = ", s.betas[-1].item())
    # print("alpha_bar_1 = ", s.alpha_bar[0].item())
    # print("alpha_bar_1000 = ", s.alpha_bar[-1].item())

    x_0 = torch.randn(5, 1, 28, 28)
    t = torch.tensor([1, 100, 300, 600, 999])
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, s.alpha_bar, noise=eps)

    print("t = 1: ", corr(x_t[0], x_0[0]))
    print("t = 100: ", corr(x_t[1], x_0[1]))
    print("t = 300: ", corr(x_t[2], x_0[2]))
    print("t = 600: ", corr(x_t[3], x_0[3]))
    print("t = 999: ", corr(x_t[4], x_0[4]))
