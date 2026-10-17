import io
import math
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Apex34P(nn.Module):
    """
    34-Parameter Gauge-Reduced Trained Qwen3 Transformer:
      - arc_start (1), arc_stride (1)                     -> 2 params (arc_A = 1 fixed by input-RMSNorm homogeneity)
      - ln_tail (2) with fixed w0 = 1.0                   -> 2 params (projective RMSNorm scale gauge)
      - q_proj (4x3 = 12), k_theta (1), q_norm (4)        -> 17 params
      - up_proj (2x3 = 6), gate_proj (2x3 = 6), down_theta (1) -> 13 params
    Total = 2 + 2 + 17 + 13 = 34 unique trainable parameters!
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
        self.gate_proj = nn.Parameter(torch.zeros(2, 3))
        self.down_theta = nn.Parameter(torch.zeros(()))

        cos, sin = precompute_rope(4, TOTAL_LEN + 1, ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.triu(torch.full((TOTAL_LEN + 1, TOTAL_LEN + 1), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _ln(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        w = torch.cat([torch.ones(1, device=x.device, dtype=x.dtype), self.ln_tail])
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

        # SwiGLU MLP block
        h2 = self._ln(x)
        up = F.linear(h2, self.up_proj)
        gate = F.linear(h2, self.gate_proj)
        act = F.silu(gate) * up

        cd, sd = torch.cos(self.down_theta), torch.sin(self.down_theta)
        a0, a1 = act[..., 0::2], act[..., 1::2]
        rot_act = torch.stack([a0 * cd - a1 * sd, a0 * sd + a1 * cd], dim=-1).flatten(-2)
        x = x + F.linear(rot_act, self.up_proj.t())

        x_norm = self._ln(x)
        return F.linear(x_norm, emb)


class Apex32P(nn.Module):
    """
    32-Parameter Trained Qwen3 Transformer:
    Adds 2 structural parameter ties on top of Apex34P:
    1. gate_proj[0, 0] = -gate_proj[0, 1]  (symmetric phase difference detector in MLP gate row 0)
    2. q_proj[3, 2] = -2.0 * q_proj[0, 0]  (coupled carry-state query projection)
    Total = 34 - 2 = 32 unique trainable parameters!
    """
    def __init__(self, teacher: Apex34P, tie_q: bool = True, tie_g: bool = True):
        super().__init__()
        self.tie_q = tie_q
        self.tie_g = tie_g
        self.arc_start = nn.Parameter(teacher.arc_start.detach().clone())
        self.arc_stride = nn.Parameter(teacher.arc_stride.detach().clone())
        self.ln_tail = nn.Parameter(teacher.ln_tail.detach().clone())

        q_flat = teacher.q_proj.detach().clone().reshape(-1)
        if tie_q:
            self.q_11 = nn.Parameter(q_flat[:11].clone())
        else:
            self.q_12 = nn.Parameter(q_flat.clone())

        self.k_theta = nn.Parameter(teacher.k_theta.detach().clone())
        self.q_norm = nn.Parameter(teacher.q_norm.detach().clone())
        self.up_proj = nn.Parameter(teacher.up_proj.detach().clone())

        g_flat = teacher.gate_proj.detach().clone().reshape(-1)
        if tie_g:
            g_init = g_flat[1:].clone()
            g_init[0] = 0.5 * (g_flat[1] - g_flat[0])
            self.gate_5 = nn.Parameter(g_init)
        else:
            self.gate_6 = nn.Parameter(g_flat.clone())

        self.down_theta = nn.Parameter(teacher.down_theta.detach().clone())

        cos, sin = precompute_rope(4, TOTAL_LEN + 1, ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.triu(torch.full((TOTAL_LEN + 1, TOTAL_LEN + 1), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _get_q(self):
        if self.tie_q:
            return torch.cat([self.q_11, -2.0 * self.q_11[0:1]]).view(4, 3)
        return self.q_12.view(4, 3)

    def _get_g(self):
        if self.tie_g:
            return torch.cat([-self.gate_5[0:1], self.gate_5]).view(2, 3)
        return self.gate_6.view(2, 3)

    def _ln(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        w = torch.cat([torch.ones(1, device=x.device, dtype=x.dtype), self.ln_tail])
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

        q_w = self._get_q()
        g_w = self._get_g()

        h = self._ln(x)
        q_raw = F.linear(h, q_w)
        ck, sk = torch.cos(self.k_theta), torch.sin(self.k_theta)
        q0, q1 = q_raw[..., 0::2], q_raw[..., 1::2]
        k_raw = torch.stack([q0 * ck - q1 * sk, q0 * sk + q1 * ck], dim=-1).flatten(-2)
        v = q_raw.view(B, T, 1, 4).transpose(1, 2)

        q = apply_rope(self._qk_norm(q_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)
        k = apply_rope(self._qk_norm(k_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)

        scores = (q @ k.transpose(-2, -1)) * 0.5 + self.causal_mask[:T, :T]
        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, 4)
        x = x + F.linear(attn_out, q_w.t())

        h2 = self._ln(x)
        up = F.linear(h2, self.up_proj)
        gate = F.linear(h2, g_w)
        act = F.silu(gate) * up

        cd, sd = torch.cos(self.down_theta), torch.sin(self.down_theta)
        a0, a1 = act[..., 0::2], act[..., 1::2]
        rot_act = torch.stack([a0 * cd - a1 * sd, a0 * sd + a1 * cd], dim=-1).flatten(-2)
        x = x + F.linear(rot_act, self.up_proj.t())

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

    teacher = Apex34P()
    with torch.no_grad():
        teacher.arc_start.copy_(sd["arc_start"])
        teacher.arc_stride.copy_(sd["arc_stride"])
        teacher.ln_tail.copy_(ln[1:] * beta)
        teacher.q_proj.copy_(sd["block.attn.q_proj.weight"] * gamma)
        teacher.k_theta.copy_(sd["block.attn.k_proj.theta"])
        teacher.q_norm.copy_(sd["block.attn.q_norm.weight"])
        teacher.up_proj.copy_(sd["block.mlp.up_proj.weight"] * delta)
        teacher.gate_proj.copy_(sd["block.mlp.gate_proj.weight"] * eta)
        teacher.down_theta.copy_(sd["block.mlp.down_proj.theta"])

    print("Apex34P params:", sum(p.numel() for p in teacher.parameters()))
    for k, v in teacher.state_dict().items():
        print(f"  {k}: {v.tolist()}")

    # Build validation set (1000 pairs from seed 2025)
    rng_val = random.Random(2025)
    val_pairs = [(rng_val.randint(0, 9999999999), rng_val.randint(0, 9999999999)) for _ in range(1000)]
    val_inps, val_tgts = zip(*(encode_pair_with_target(a, b) for a, b in val_pairs))
    val_x = torch.tensor(val_inps, dtype=torch.long)
    val_y = torch.tensor(val_tgts, dtype=torch.long)

    @torch.inference_mode()
    def eval_model(m):
        logits = m(val_x)[:, INPUT_LEN - 1 :, :]
        preds = logits.argmax(dim=-1)
        tok_acc = (preds == val_y).float().mean().item() * 100
        seq_acc = (preds == val_y).all(dim=-1).float().mean().item() * 100
        return tok_acc, seq_acc

    t_tok, t_seq = eval_model(teacher)
    print(f"Apex34P Val (1000 pairs): tok={t_tok:.3f}%, seq={t_seq:.2f}%")

    # Build distillation training set (1500 pairs)
    rng_tr = random.Random(101)
    tr_pairs = [(rng_tr.randint(0, 9999999999), rng_tr.randint(0, 9999999999)) for _ in range(1500)]
    tr_inps, tr_tgts = zip(*(encode_pair_with_target(a, b) for a, b in tr_pairs))
    tr_x = torch.tensor(tr_inps, dtype=torch.long)
    tr_y = torch.tensor(tr_tgts, dtype=torch.long)
    with torch.no_grad():
        t_logits = teacher(tr_x)[:, INPUT_LEN - 1 :, :].detach()

    for tie_q, tie_g in [(True, False), (False, True), (True, True)]:
        student = Apex32P(teacher, tie_q=tie_q, tie_g=tie_g)
        n_p = sum(p.numel() for p in student.parameters())
        tok0, seq0 = eval_model(student)
        print(f"\n--- Student {n_p}p (tie_q={tie_q}, tie_g={tie_g}) Initial: tok={tok0:.2f}%, seq={seq0:.2f}% ---")

        # Freeze arc_start and arc_stride during L-BFGS logit distillation
        student.arc_start.requires_grad_(False)
        student.arc_stride.requires_grad_(False)
        trainable = [p for p in student.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(trainable, lr=0.5, max_iter=50, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            s_logits = student(tr_x)[:, INPUT_LEN - 1 :, :]
            loss = F.mse_loss(s_logits, t_logits)
            loss.backward()
            return loss

        for step in range(4):
            loss = opt.step(closure)
            tok1, seq1 = eval_model(student)
            print(f"  L-BFGS Step {step+1}: mse={loss.item():.6f}, tok={tok1:.3f}%, seq={seq1:.2f}%")
            if seq1 == 100.0:
                torch.save(student.state_dict(), f"distilled_{n_p}p_q{int(tie_q)}_g{int(tie_g)}.pt")
                for k, v in student.state_dict().items():
                    print(f"    {k}: {v.tolist()}")
                break


if __name__ == "__main__":
    main()
