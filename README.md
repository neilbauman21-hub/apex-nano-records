# APEX // NANO-RECORDS

**Four Verified #1 Parameter-Efficiency & Byte-Level Language Model Benchmark Records**

- **Live Interactive Monograph & Verifier**: [https://neilbauman21-hub.github.io/apex-nano-records/](https://neilbauman21-hub.github.io/apex-nano-records/)
- **Hugging Face Hub (Weights & Proofs)**: [https://huggingface.co/e12ex2/apex-nano-records](https://huggingface.co/e12ex2/apex-nano-records)
- **Hugging Face Live Space**: [https://huggingface.co/spaces/e12ex2/apex-nano-records](https://huggingface.co/spaces/e12ex2/apex-nano-records)

---

## Summary of Verified #1 Records

| Benchmark / Track | Previous #1 Record | APEX // NANO-RECORDS (#1) | Improvement | Verification |
| :--- | :--- | :--- | :--- | :--- |
| **AdderBoard Constructive Track** (10-digit addition) | `6* params` (`zcbtrak`) | **`4 params`** (`submissions/submission_4p_record.py`) | **-33.3% parameters** (`10,010 / 10,010` = `100.00%`) | `verify_output_4p_record.txt` |
| **AdderBoard Strict `nn.Parameter` Track** (10-digit addition) | `8 params` (`kswain98`) | **`6 params`** (`submissions/submission_6p_strict_record.py`) | **-25.0% parameters** (`10,010 / 10,010` = `100.00%`) | `verify_output_6p_strict_record.txt` |
| **AdderBoard Trained Track** (10-digit addition, gradient-trained) | `36 params` (`tbukic M10S-36p`) | **`28 params`** (`submissions/submission_28p_trained_record.py`) & **`34 params`** (`submissions/submission_34p_trained_record.py`) | **-22.2% parameters** (`10,010 / 10,010` = `100.00%`, breaks 30p barrier) | `verify_output_28p_trained_record.txt` |
| **Tiny800 Byte-Level LM** (`800` scalar params, `seed=42` frozen codebooks) | `3.9812 BPB` (`Tiny800 Stage C`, 1B tokens / 22.9 min) | **`3.6935 BPB`** (`Omega-800`, `omega800/train_and_verify_omega800.py`) | **-0.2877 bits/byte** in **77.0s** (`19.7M` tokens) | `omega800/omega800_results.json` |

---

## 1. AdderBoard #1 Records (4p Constructive, 6p Strict, 28p Trained)

### Mathematical Core

1. **4-Parameter Constructive Record (`submissions/submission_4p_record.py`)**:
   - **Coupled Reciprocal Parabolic Embedding**: Eliminates separate parabolic curvature and carry-scale parameters by tying the quadratic embedding scale directly to the reciprocal of the attention temperature $w_0 = 1000.0$:
     $$e(d) = \left[w_0 - \frac{d^2}{w_0},\; -d\right]$$
   - **Unit-Residual SwiGLU Carry Cascade**: Normalizes the carry recurrence so the residual stream directly transports $c_i \in \{0, 1\}$ with unit coefficient, requiring only 4 unique scalar parameters: `{1000.0, 22.0, -10.0, 18.5}`.

2. **6-Parameter Strict `nn.Parameter` Record (`submissions/submission_6p_strict_record.py`)**:
   - Stores all tensor projection matrices as explicit `nn.Parameter` modules (`q_proj_weight`, `v_proj_weight`, `gate_up_weight`, `down_proj_weight`, `lm_head_weight`) with zero Python magic numbers inside `forward()`, reducing strict parameter count from `8` (`kswain98`) to **`6`**.

3. **28-Parameter Trained Record (`submissions/submission_28p_trained_record.py`)**:
   - **Exact Continuous Gauge Reduction ($36 \to 34$)**: Absorbs the two RMSNorm gain parameters `(g0, g1)` into `(q_proj, k_proj)` and `v_proj` via scale invariance of RMSNorm (`submissions/submission_34p_trained_record.py`).
   - **Coupled Query-Subspace & Spiral Phase Tie ($34 \to 28$)**: Parameterizes the 8-element query projection inside a 2-parameter rank-1 phase basis and ties the spiral positional frequency to the token embedding curvature via L-BFGS + Coordinate Pattern Search distillation on carry-chain boundaries, achieving **10,010 / 10,010 (100.00%)** with **28 unique parameters**.

### Reproduction

```bash
python3 verify.py submissions/submission_4p_record.py --seed 2025
python3 verify.py submissions/submission_6p_strict_record.py --seed 2025
python3 verify.py submissions/submission_34p_trained_record.py --seed 2025
python3 verify.py submissions/submission_28p_trained_record.py --seed 2025
```

---

## 2. Tiny800 #1 Byte-Level Language Model Record (`Omega-800`: `3.6935 BPB`)

### Mathematical Core: Full-Rank Kronecker Tensor-Product Codebook Readout

Standard `Tiny800` models project their hidden state through a single frozen output codebook $C_{\text{out}} \in \mathbb{R}^{256 \times 30}$ (`seed=42`), which caps the output logit subspace at $\text{rank}(C_{\text{out}}) = 30 < 256$—creating a hard linear bottleneck across the 256-byte vocabulary.

`Omega-800` uses the **exact same `seed=42` frozen codebooks** ($C_{\text{in}} \in \mathbb{R}^{256 \times 8}, C_{\text{out}} \in \mathbb{R}^{256 \times 30}$) without adding a single frozen or learned vocabulary parameter. By forming the bilinear Kronecker interaction between the input and output codebooks:

$$\text{logits}(b) = \sum_{i=0}^{8} \sum_{j=0}^{29} \tilde{C}_{\text{in}}[b, i]\, M_{i,j}(x_t)\, C_{\text{out}}[b, j], \quad \tilde{C}_{\text{in}} = [1, C_{\text{in}}] \in \mathbb{R}^{256 \times 9}$$

the induced $256 \times 270$ readout basis $\Psi = \tilde{C}_{\text{in}} \odot_{\text{row}} C_{\text{out}}$ achieves **full column rank**:

$$\text{rank}(\Psi) = 256$$

Combined with a 15-tap causal multi-scale exponential memory bank ($530 + 270 = 800$ exact learned scalar parameters), `Omega-800` reaches **`3.6935 BPB`** on `val_curriculum_v1.bin` in **77.0 seconds** on Apple M4 (`19.7M` tokens), surpassing the 1-billion-token `Tiny800 Stage C` record (`3.9812 BPB`) by **0.2877 bits/byte**.

### Reproduction

```bash
python3 omega800/train_and_verify_omega800.py
```

---

## Visual Preview

![APEX // NANO-RECORDS Monograph](screenshot-1-monograph-4p.png)
![28-Parameter Trained Record](screenshot-2-trained-28p.png)
