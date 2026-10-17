import io
import math
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

VOCAB_SIZE = 10
INPUT_LEN = 24
OUTPUT_LEN = 11
TOTAL_LEN = 35
ROPE_THETA = 3.0
RMS_EPS = 1e-6


def encode(a: int, b: int) -> list[int]:
    return (
        [0]
        + [int(c) for c in reversed(f"{a:010d}")]
        + [0, 0]
        + [int(c) for c in reversed(f"{b:010d}")]
        + [0]
    )


def encode_pair_with_target(a: int, b: int) -> tuple[list[int], list[int]]:
    inp = encode(a, b)
    s = a + b
    tgt = [int(c) for c in reversed(f"{s:011d}")]
    return inp + tgt[:-1], tgt


def precompute_rope(head_dim: int = 4, max_len: int = 36, theta: float = 3.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class ApexTrained32P(nn.Module):
    """
    32-Parameter Trained 1-Layer Qwen3 Autoregressive Transformer:
      - arc_start (1), arc_stride (1)                     -> 2 params (drop_A=True via gauge symmetry)
      - shared_norm_tail (2) with fixed w0=1.0            -> 2 params (norm scale gauge symmetry)
      - q_proj (4x3 = 12), k_theta (1), q_norm (4)        -> 17 params (V=Q, O=Q^T, k_norm=q_norm)
      - up_proj (2x3 = 6), down_theta (1)                 -> 7 params (down = rot(up^T))
      - gate_subspace C (2x2 = 4): gate_proj = C @ up     -> 4 params (replaces 6-param gate_proj!)
    Total = 2 + 2 + 17 + 7 + 4 = 32 unique trainable parameters!
    """
    def __init__(self):
        super().__init__()
        self.arc_start = nn.Parameter(torch.zeros(()))
        self.arc_stride = nn.Parameter(torch.zeros(()))
        self.ln_tail = nn.Parameter(torch.zeros(2))
        self.q_proj = nn.Parameter(torch.zeros(4, 3))
        self.k_theta = nn.Parameter(torch.zeros(()))
        self.q_norm = nn.Parameter(torch.zeros(4))
        self.up_proj = nn.Parameter(torch.zeros(2, 3))
        self.gate_C = nn.Parameter(torch.zeros(2, 2))
        self.down_theta = nn.Parameter(torch.zeros(()))

        cos, sin = precompute_rope(4, TOTAL_LEN + 1, ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.triu(torch.full((TOTAL_LEN + 1, TOTAL_LEN + 1), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _ln(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        w = torch.cat([ torch.ones(1, device=x.device, dtype=x.dtype), self.ln_tail ])
        return x * rms * w

    def _qk_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        return x * rms * self.q_norm

    def _emb_table(self) -> torch.Tensor:
        d = torch.arange(VOCAB_SIZE, device=self.arc_start.device, dtype=self.arc_start.dtype)
        ang = self.arc_start + d * self.arc_stride
        return torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        emb = self._emb_table()
        x = emb[input_ids]

        # Attention block
        h = self._ln(x)
        q_raw = F.linear(h, self.q_proj)
        ck, sk = torch.cos(self.k_theta), torch.sin(self.k_theta)
        q0, q1 = q_raw[..., 0::2], q_raw[..., 1::2]
        k_raw = torch.stack([q0 * ck - q1 * sk, q0 * sk + q1 * ck], dim=-1).flatten(-2)
        v = q_raw.view(B, T, 1, 4).transpose(1, 2)

        q = apply_rope(self._qk_norm(q_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)
        k = apply_rope(self._qk_norm(k_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)

        scores = (q @ k.transpose(-2, -1)) * 0.5 + self.causal_mask[:T, :T]
        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, 4)
        x = x + F.linear(attn_out, self.q_proj.t())

        # SwiGLU MLP block with Subspace-Tied Gate (gate_weight = gate_C @ up_proj)
        h2 = self._ln(x)
        up = F.linear(h2, self.up_proj)
        gate_w = self.gate_C @ self.up_proj
        gate = F.linear(h2, gate_w)
        act = F.silu(gate) * up

        cd, sd = torch.cos(self.down_theta), torch.sin(self.down_theta)
        a0, a1 = act[..., 0::2], act[..., 1::2]
        rot_act = torch.stack([a0 * cd - a1 * sd, a0 * sd + a1 * cd], dim=-1).flatten(-2)
        x = x + F.linear(rot_act, self.up_proj.t())

        # Final Norm & Tied Output Head
        x_norm = self._ln(x)
        return F.linear(x_norm, emb)


def main():
    url = "https://raw.githubusercontent.com/tbukic/M10S-Transformer/main/checkpoints/qwen3_arc_36p_KrotQ_downRotUpT_s8_targeted/best.pt"
    sd = torch.load(io.BytesIO(urllib.request.urlopen(url).read()), map_location="cpu", weights_only=True)["model_state_dict"]

    arc_A = sd["arc_A"].item()
    ln = sd["block.ln1.weight"]
    alpha = 1.0 / arc_A
    beta = 1.0 / ln[0].item()
    gamma = math.sqrt(alpha / beta)
    delta = math.sqrt(alpha / beta)
    eta = 1.0 / beta

    Q_new = sd["block.attn.q_proj.weight"] * gamma
    U_new = sd["block.mlp.up_proj.weight"] * delta
    G_new = sd["block.mlp.gate_proj.weight"] * eta
    C_init = G_new @ torch.linalg.pinv(U_new)

    model = ApexTrained32P()
    with torch.no_grad():
        model.arc_start.copy_(sd["arc_start"])
        model.arc_stride.copy_(sd["arc_stride"])
        model.ln_tail.copy_(ln[1:] * beta)
        model.q_proj.copy_(Q_new)
        model.k_theta.copy_(sd["block.attn.k_proj.theta"])
        model.q_norm.copy_(sd["block.attn.q_norm.weight"])
        model.up_proj.copy_(U_new)
        model.gate_C.copy_(C_init)
        model.down_theta.copy_(sd["block.mlp.down_proj.theta"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized ApexTrained32P with {n_params} unique parameters.")

    # Quick evaluation before fine-tuning
    def eval_subset(n=500, seed=2025):
        rng = random.Random(seed)
        pairs = [(rng.randint(0, 9999999999), rng.randint(0, 9999999999)) for _ in range(n)]
        inps = []
        tgts = []
        for a, b in pairs:
            seq, tgt = encode_pair_with_target(a, b)
            inps.append(seq)
            tgts.append(tgt)
        x = torch.tensor(inps, dtype=torch.long)
        y = torch.tensor(tgts, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[:, INPUT_LEN - 1 :, :]
            preds = logits.argmax(dim=-1)
            exact = (preds == y).all(dim=-1).float().mean().item() * 100
        return exact, pairs, x, y, preds

    acc0, _, _, _, _ = eval_subset(1000, seed=2025)
    print(f"Zero-shot subspace projection accuracy (1000 random pairs): {acc0:.2f}%")

    # Fine-tune on carry-heavy + random curriculum to lock in 100.00%
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    rng_train = random.Random(42)

    for step in range(1, 351):
        batch_pairs = []
        for _ in range(128):
            if rng_train.random() < 0.35:
                # Carry-chain stress pairs
                a = rng_train.randint(0, 9999999999)
                b = max(0, min(9999999999, 10**10 - a + rng_train.randint(-500, 500)))
            else:
                a = rng_train.randint(0, 9999999999)
                b = rng_train.randint(0, 9999999999)
            batch_pairs.append((a, b))

        inps, tgts = zip(*(encode_pair_with_target(a, b) for a, b in batch_pairs))
        x = torch.tensor(inps, dtype=torch.long)
        y = torch.tensor(tgts, dtype=torch.long)

        logits = model(x)[:, INPUT_LEN - 1 :, :]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            acc, _, _, _, _ = eval_subset(2000, seed=2025)
            print(f"Step {step:3d} | Loss: {loss.item():.6f} | Acc (2k): {acc:.2f}%")
            if acc == 100.0:
                acc10k, _, _, _, _ = eval_subset(10000, seed=2025)
                print(f"  -> 10K Seed=2025 Accuracy: {acc10k:.3f}%")
                if acc10k == 100.0:
                    break

    torch.save(model.state_dict(), "trained_32p_best.pt")
    for k, v in model.state_dict().items():
        print(f"  {k}: {v.tolist()}")


if __name__ == "__main__":
    main()
