import torch


def time_embedding(t, dim=128):
    assert dim % 2 == 0, "dim 必须是偶数"
    if type(t) == int:
        t = torch.tensor([t])
    i = torch.arange(0, dim / 2)
    freqs = 1 / 10000 ** (i * 2 / dim)
    broadcast = torch.outer(t, freqs)
    emb = torch.zeros(broadcast.shape[0], dim)
    emb[:, 0::2] = torch.sin(broadcast)
    emb[:, 1::2] = torch.cos(broadcast)
    return emb


def cosine_sim_matrix(emb):
    norm = emb / emb.norm(dim=1, keepdim=True)
    sim = norm @ norm.T
    return sim


if __name__ == "__main__":
    t = torch.tensor([1, 10, 100, 300, 600, 999])
    emb = time_embedding(t, dim=128)
    print(emb.shape)

    sim = cosine_sim_matrix(emb)
    print(sim)
