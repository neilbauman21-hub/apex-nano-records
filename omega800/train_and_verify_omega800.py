"""
Train and verify Omega-800 on the official Tiny800 dataset (`train_curriculum_v1.bin` and `val_curriculum_v1.bin`),
measuring held-out Bits Per Byte (BPB) and Next-Byte Top-1 Accuracy.
"""

import math
import time
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TINY800_DIR = Path("/Users/neilthepineapplegoat/.gemini/antigravity/scratch/tiny800")
sys.path.insert(0, str(TINY800_DIR))

from model import generate_deterministic_codebooks
from evaluate import evaluate_corpus


class Omega800KroneckerLM(nn.Module):
    """
    Strictly 800-Parameter Full-Rank Kronecker Tensor-Field & SwiGLU Byte Language Model.

    Operates in the 270-dimensional Kronecker product space `R^{9 x 30}` formed by:
      `Psi(b) = [1, C_in[b, 0..7]]^T (x) C_out[b, 0..29]  in R^{9 x 30}` (270 dimensions > 256 bytes).
    Because `rank(Psi) = 256`, every byte `b in [0..255]` has a unique representation in `R^{9 x 30}`
    using ONLY the frozen `Tiny800` buffers `input_codebook (256x8)` and `output_codebook (256x30)`.

    Exact Parameter Breakdown (Total = 800 Scalar Parameters):
      1. Dynamic Kronecker Transition Operators `T_k = (A_k @ Psi @ R_k^T) @ Psi^T` in `R^{256 x 256}`:
         - `A_lag`:     (3, 9, 9)   = 243 params  (Left Mode-9 operators for lags 0, 1, 2)
         - `R_u`:       (3, 30, 2)  = 180 params  (Right Mode-30 low-rank factors)
         - `R_v`:       (3, 30, 2)  = 180 params  (Right Mode-30 low-rank factors)
         - `R_diag`:    (3, 30)     =  90 params  (Right Mode-30 full-rank diagonal scalings)
      2. Nonlinear Multi-Horizon SwiGLU Context Core (reads 38D `[C_in, C_out]` across 8 causal lags):
         - `W_ctx`:     (2, 38)     =  76 params  (Projects 38D dual codebook to 2D gating/modulation coordinates)
         - `W_horiz`:   (3, 8)      =  24 params  (Weights 8 causal horizons: lags 0..5 + 2 EMA traces)
         - `scales`:    (7,)        =   7 params  (Logit & context modulation temperatures)
      -------------------------------------------------------------------------
      Total = 243 + 180 + 180 + 90 + 76 + 24 + 7 = 800 trainable scalar parameters!
    """
    def __init__(self, codebook_seed: int = 42):
        super().__init__()
        c_in, c_out = generate_deterministic_codebooks(
            vocab_size=256,
            codebook_in_dim=8,
            output_dim=30,
            seed=codebook_seed,
        )
        self.register_buffer("input_codebook", c_in)    # (256, 8)
        self.register_buffer("output_codebook", c_out)  # (256, 30)

        # 1. Dynamic Kronecker Transition Operators in R^{9 x 30}: 243 + 180 + 180 + 90 = 693 params
        self.A_lag = nn.Parameter(torch.randn(3, 9, 9) * 0.15)
        with torch.no_grad():
            for i in range(3):
                self.A_lag[i].add_(torch.eye(9) * 0.3)

        self.R_u = nn.Parameter(torch.randn(3, 30, 2) * 0.15)
        self.R_v = nn.Parameter(torch.randn(3, 30, 2) * 0.15)
        self.R_diag = nn.Parameter(torch.ones(3, 30) * 0.4)

        # 2. Nonlinear Multi-Horizon Context Core: 76 + 24 + 7 = 107 params
        self.W_ctx = nn.Parameter(torch.randn(2, 38) * 0.15)
        init_h = torch.tensor([
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.4, 0.1, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.5, 0.6, 0.5, 0.4, 0.6, 0.8],
        ], dtype=torch.float32)
        self.W_horiz = nn.Parameter(init_h)
        self.scales = nn.Parameter(torch.tensor([6.0, 4.0, 2.5, 1.5, 1.0, 0.5, 0.0]))

        self._verify_parameter_count()

    def _verify_parameter_count(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert total == 800 and trainable == 800, f"Expected 800 params, got {total}"
        return total

    def _get_psi_vocab(self) -> torch.Tensor:
        ones = torch.ones(256, 1, device=self.input_codebook.device, dtype=self.input_codebook.dtype)
        c9 = torch.cat([ones, self.input_codebook], dim=-1)  # (256, 9)
        return c9.unsqueeze(2) * self.output_codebook.unsqueeze(1)  # (256, 9, 30)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        idx = x.long()
        B, T = idx.shape

        # 1. Precompute the 3 Full-Rank 256x256 Kronecker Transition Matrices on the 256-entry vocabulary!
        #    Cost is O(256^2 * 270) regardless of batch size B*T!
        psi = self._get_psi_vocab()         # (256, 9, 30)
        psi_flat = psi.view(256, 270)       # (256, 270)

        T_mats = []
        for k in range(3):
            Rk = torch.diag(self.R_diag[k]) + self.R_u[k] @ self.R_v[k].t()  # (30, 30)
            # Transform 256 vocabulary entries in R^{9 x 30}
            psi_k = torch.matmul(torch.matmul(self.A_lag[k], psi), Rk.t()).view(256, 270)
            # Contract with Psi^T -> (256, 256) byte-to-byte transition table
            T_mats.append(psi_k @ psi_flat.t())

        # Causal shifted byte indices for lags 0, 1, 2, 3, 4, 5
        idx0 = idx
        idx1 = F.pad(idx[:, :-1], (1, 0), value=32) if T > 1 else idx
        idx2 = F.pad(idx[:, :-2], (2, 0), value=32) if T > 2 else idx
        idx3 = F.pad(idx[:, :-3], (3, 0), value=32) if T > 3 else idx
        idx4 = F.pad(idx[:, :-4], (4, 0), value=32) if T > 4 else idx
        idx5 = F.pad(idx[:, :-5], (5, 0), value=32) if T > 5 else idx

        # Fast table lookups for lag-0, lag-1, lag-2 transitions + longer-horizon smoothed transitions
        L0 = T_mats[0][idx0]  # (B, T, 256)
        L1 = T_mats[1][idx1]  # (B, T, 256)
        L2 = T_mats[2][idx2] + 0.5 * T_mats[2][idx3] + 0.25 * T_mats[2][idx4]  # (B, T, 256)

        # 2. Nonlinear Multi-Horizon Context Gating (modulates interaction between L0, L1, L2)
        feat38_vocab = torch.cat([self.input_codebook, self.output_codebook], dim=-1)  # (256, 38)
        c2_vocab = F.linear(feat38_vocab, self.W_ctx)  # (256, 2)

        u0 = c2_vocab[idx0]
        u1 = c2_vocab[idx1]
        u2 = c2_vocab[idx2]
        u3 = c2_vocab[idx3]
        u4 = c2_vocab[idx4]
        u5 = c2_vocab[idx5]
        ema_f = 0.5 * u0 + 0.3 * u1 + 0.2 * u2
        ema_s = 0.25 * (u2 + u3 + u4 + u5)

        taps = torch.stack([u0, u1, u2, u3, u4, u5, ema_f, ema_s], dim=-1)  # (B, T, 2, 8)
        h_ctx = torch.matmul(taps, self.W_horiz.t())  # (B, T, 2, 3)

        g0 = torch.sigmoid(h_ctx[:, :, 0, 0:1] + self.scales[4])
        g1 = torch.sigmoid(h_ctx[:, :, 0, 1:2] + self.scales[5])
        g2 = F.silu(h_ctx[:, :, 1, 2:3] + self.scales[6])

        # Combine additive + multiplicative (bigram/trigram intersection) Kronecker logits!
        logits = (
            self.scales[0] * g0 * L0
            + self.scales[1] * g1 * L1
            + self.scales[2] * L2
            + self.scales[3] * g2 * (L0 * torch.sigmoid(L1))
        )

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.long().reshape(-1))

        return logits, loss, h_ctx[:, -1, :, 0].detach()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_bin = np.memmap(TINY800_DIR / "data" / "generated" / "train_curriculum_v1.bin", dtype=np.uint8, mode="r")
    val_bin = np.memmap(TINY800_DIR / "data" / "generated" / "val_curriculum_v1.bin", dtype=np.uint8, mode="r")
    train_tensor = torch.from_numpy(np.array(train_bin, copy=True))
    val_tensor = torch.from_numpy(np.array(val_bin, copy=True))
    print(f"Loaded train.bin ({len(train_tensor):,} bytes) and val.bin ({len(val_tensor):,} bytes)", flush=True)

    model = Omega800KroneckerLM(codebook_seed=42).to(device)
    n_params = model._verify_parameter_count()
    print(f"Verified Omega800KroneckerLM parameter count: {n_params} scalar parameters", flush=True)

    init_metrics = evaluate_corpus(model, val_tensor, batch_size=256, seq_len=256, device=device)
    print(f"Step 0 (Init) | Val BPB: {init_metrics['val_bpb']:.4f} | Val Acc: {init_metrics['val_acc']*100:.2f}%", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-4)
    total_steps = 600
    batch_size = 128
    seq_len = 256
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-4)

    t0 = time.time()
    n_train = len(train_tensor) - seq_len - 2
    rng = torch.Generator().manual_seed(2026)

    best_bpb = 999.0
    history = [{
        "step": 0,
        "tokens_millions": 0.0,
        "val_bpb": round(init_metrics["val_bpb"], 4),
        "val_acc_pct": round(init_metrics["val_acc"] * 100, 2),
        "elapsed_sec": 0.0,
    }]

    for step in range(1, total_steps + 1):
        starts = torch.randint(0, n_train, (batch_size,), generator=rng)
        offsets = torch.arange(seq_len + 1).unsqueeze(0)
        chunks = train_tensor[starts.unsqueeze(1) + offsets].to(device)
        x = chunks[:, :-1]
        y = chunks[:, 1:]

        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == total_steps:
            metrics = evaluate_corpus(model, val_tensor, batch_size=256, seq_len=256, device=device)
            elapsed = time.time() - t0
            tokens_seen = step * batch_size * seq_len
            print(
                f"Step {step:4d}/{total_steps} ({tokens_seen/1e6:5.1f}M tok, {elapsed:5.1f}s) | "
                f"Val BPB: {metrics['val_bpb']:.4f} | Val Acc: {metrics['val_acc']*100:.2f}%",
                flush=True,
            )
            history.append({
                "step": step,
                "tokens_millions": round(tokens_seen / 1e6, 2),
                "val_bpb": round(metrics["val_bpb"], 4),
                "val_acc_pct": round(metrics["val_acc"] * 100, 2),
                "elapsed_sec": round(elapsed, 2),
            })
            if metrics["val_bpb"] < best_bpb:
                best_bpb = metrics["val_bpb"]
                torch.save(model.state_dict(), Path(__file__).parent / "omega800_best.pt")

    out_json = Path(__file__).parent / "omega800_results.json"
    out_json.write_text(json.dumps({
        "model": "Omega-800 Kronecker Dual-Codebook Tensor-Field Byte LM",
        "trainable_params": n_params,
        "tiny800_stage_c_1B_baseline": {"val_bpb": 3.9812, "val_acc_pct": 45.52, "tokens": 1_000_013_824},
        "omega800_final": history[-1],
        "history": history,
    }, indent=2))
    print(f"Saved results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
