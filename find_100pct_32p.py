import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from distill_32p_lbfgs import Apex34P, encode_pair_with_target, precompute_rope, apply_rope, VOCAB_SIZE, INPUT_LEN, TOTAL_LEN, ROPE_THETA, RMS_EPS


class Apex32P_Q56_G01(nn.Module):
    """
    32-Parameter Trained Qwen3 Transformer:
    Uses two structural ties on top of the 34-param gauge-reduced model:
      1. gate_proj[0, 0] = -gate_proj[0, 1]  (symmetric phase difference detector)
      2. q_proj[1, 2] = -2.0 * q_proj[2, 0]  (since q_flat[5] = 1.300808 vs -2 * q_flat[6] = 1.302002, within 0.09%!)
    """
    def __init__(self, sd33: dict, also_tie_q11: bool = False):
        super().__init__()
        self.also_tie_q11 = also_tie_q11
        self.arc_start = nn.Parameter(sd33["arc_start"].clone())
        self.arc_stride = nn.Parameter(sd33["arc_stride"].clone())
        self.ln_tail = nn.Parameter(sd33["ln_tail"].clone())

        q12 = sd33["q_12"].clone()
        # Tie index 5 to -2.0 * index 6
        q12[6] = 0.5 * (q12[6] - 0.5 * q12[5])
        if also_tie_q11:
            # 31 parameters total! Tie index 11 to -2.0 * index 0 AND index 5 to -2.0 * index 6
            # Keep indices [0, 1, 2, 3, 4, 6, 7, 8, 9, 10] -> 10 params in Q!
            keep_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
            self.q_free = nn.Parameter(q12[keep_idx].clone())
        else:
            # 32 parameters total! Keep indices [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11] -> 11 params in Q!
            keep_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
            self.q_free = nn.Parameter(q12[keep_idx].clone())

        self.k_theta = nn.Parameter(sd33["k_theta"].clone())
        self.q_norm = nn.Parameter(sd33["q_norm"].clone())
        self.up_proj = nn.Parameter(sd33["up_proj"].clone())
        self.gate_5 = nn.Parameter(sd33["gate_5"].clone())
        self.down_theta = nn.Parameter(sd33["down_theta"].clone())

        cos, sin = precompute_rope(4, TOTAL_LEN + 1, ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.triu(torch.full((TOTAL_LEN + 1, TOTAL_LEN + 1), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _get_q(self):
        qf = self.q_free
        q5 = -2.0 * qf[5:6]  # since index 6 in original q12 is at index 5 in q_free!
        if self.also_tie_q11:
            q11 = -2.0 * qf[0:1]
            full_q = torch.cat([qf[:5], q5, qf[5:], q11])
        else:
            full_q = torch.cat([qf[:5], q5, qf[5:]])
        return full_q.view(4, 3)

    def _get_g(self):
        return torch.cat([-self.gate_5[0:1], self.gate_5]).view(2, 3)

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
    sd33 = torch.load("distilled_33p_q0_g1.pt", map_location="cpu", weights_only=True)

    rng_val = random.Random(2025)
    val_pairs = [(rng_val.randint(0, 9999999999), rng_val.randint(0, 9999999999)) for _ in range(3000)]
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

    # Build reference teacher from 33p (which has 100.00% accuracy)
    from distill_32p_lbfgs import Apex32P, Apex34P
    t33 = Apex32P(Apex34P(), tie_q=False, tie_g=True)
    t33.load_state_dict(sd33)
    t33.eval()

    rng_tr = random.Random(101)
    tr_pairs = [(rng_tr.randint(0, 9999999999), rng_tr.randint(0, 9999999999)) for _ in range(1500)]
    tr_inps, tr_tgts = zip(*(encode_pair_with_target(a, b) for a, b in tr_pairs))
    tr_x = torch.tensor(tr_inps, dtype=torch.long)
    with torch.no_grad():
        t_logits = t33(tr_x)[:, INPUT_LEN - 1 :, :].detach()

    for also_31p in [False, True]:
        m = Apex32P_Q56_G01(sd33, also_tie_q11=also_31p)
        n_p = sum(p.numel() for p in m.parameters())
        tok0, seq0 = eval_model(m)
        print(f"\n=== Model {n_p}p (also_tie_q11={also_31p}) Zero-Shot: tok={tok0:.4f}%, seq={seq0:.2f}% ===")
        if seq0 == 100.0:
            torch.save(m.state_dict(), f"best_{n_p}p.pt")
            for k, v in m.state_dict().items():
                print(f"  {k}: {v.tolist()}")

        m.arc_start.requires_grad_(False)
        m.arc_stride.requires_grad_(False)
        opt = torch.optim.LBFGS([p for p in m.parameters() if p.requires_grad], lr=0.5, max_iter=40, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            s_logits = m(tr_x)[:, INPUT_LEN - 1 :, :]
            loss = F.mse_loss(s_logits, t_logits)
            loss.backward()
            return loss
        for step in range(3):
            loss = opt.step(closure)
            tok1, seq1 = eval_model(m)
            print(f"  Step {step+1}: mse={loss.item():.7f}, tok={tok1:.4f}%, seq={seq1:.2f}%")
            if seq1 == 100.0:
                torch.save(m.state_dict(), f"best_{n_p}p.pt")
                for k, v in m.state_dict().items():
                    print(f"  {k}: {v.tolist()}")
                break


if __name__ == "__main__":
    main()
