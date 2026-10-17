import random
import torch
import torch.nn.functional as F
from distill_32p_lbfgs import Apex34P, Apex32P, encode_pair_with_target, INPUT_LEN

def main():
    teacher = Apex34P()
    sd33 = torch.load("distilled_33p_q0_g1.pt", map_location="cpu", weights_only=True)
    # Reconstruct teacher from 34p
    from distill_32p_lbfgs import urllib, io, math
    url = "https://raw.githubusercontent.com/tbukic/M10S-Transformer/main/checkpoints/qwen3_arc_36p_KrotQ_downRotUpT_s8_targeted/best.pt"
    sd = torch.load(io.BytesIO(urllib.request.urlopen(url).read()), map_location="cpu", weights_only=True)["model_state_dict"]
    arc_A = sd["arc_A"].item()
    ln = sd["block.ln1.weight"]
    alpha = 1.0 / arc_A
    beta = 1.0 / ln[0].item()
    gamma = math.sqrt(alpha / beta)
    delta = math.sqrt(alpha / beta)
    eta = 1.0 / beta
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

    student = Apex32P(teacher, tie_q=True, tie_g=True)

    rng_val = random.Random(2025)
    val_pairs = [(rng_val.randint(0, 9999999999), rng_val.randint(0, 9999999999)) for _ in range(2000)]
    val_inps, val_tgts = zip(*(encode_pair_with_target(a, b) for a, b in val_pairs))
    val_x = torch.tensor(val_inps, dtype=torch.long)
    val_y = torch.tensor(val_tgts, dtype=torch.long)

    rng_tr = random.Random(101)
    tr_pairs = [(rng_tr.randint(0, 9999999999), rng_tr.randint(0, 9999999999)) for _ in range(2000)]
    tr_inps, tr_tgts = zip(*(encode_pair_with_target(a, b) for a, b in tr_pairs))
    tr_x = torch.tensor(tr_inps, dtype=torch.long)
    tr_y = torch.tensor(tr_tgts, dtype=torch.long)
    with torch.no_grad():
        t_logits = teacher(tr_x)[:, INPUT_LEN - 1 :, :].detach()

    @torch.inference_mode()
    def eval_model(m):
        logits = m(val_x)[:, INPUT_LEN - 1 :, :]
        preds = logits.argmax(dim=-1)
        tok_acc = (preds == val_y).float().mean().item() * 100
        seq_acc = (preds == val_y).all(dim=-1).float().mean().item() * 100
        return tok_acc, seq_acc

    # Phase 1: Frozen arc
    student.arc_start.requires_grad_(False)
    student.arc_stride.requires_grad_(False)
    opt1 = torch.optim.LBFGS([p for p in student.parameters() if p.requires_grad], lr=0.5, max_iter=60, history_size=30, line_search_fn="strong_wolfe")
    def closure1():
        opt1.zero_grad()
        s_logits = student(tr_x)[:, INPUT_LEN - 1 :, :]
        loss = F.mse_loss(s_logits, t_logits)
        loss.backward()
        return loss
    for step in range(4):
        loss = opt1.step(closure1)
        t_acc, s_acc = eval_model(student)
        print(f"Phase 1 Step {step+1}: mse={loss.item():.7f}, tok={t_acc:.3f}%, seq={s_acc:.2f}%")

    # Phase 2: Unfreeze all 32 parameters and refine with L-BFGS in float64 for high precision!
    student.double()
    tr_x_d = tr_x
    t_logits_d = teacher.double()(tr_x_d)[:, INPUT_LEN - 1 :, :].detach()
    student.arc_start.requires_grad_(True)
    student.arc_stride.requires_grad_(True)
    opt2 = torch.optim.LBFGS(student.parameters(), lr=0.5, max_iter=80, history_size=40, tolerance_grad=1e-11, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure2():
        opt2.zero_grad()
        s_logits = student(tr_x_d)[:, INPUT_LEN - 1 :, :]
        loss = F.mse_loss(s_logits, t_logits_d)
        loss.backward()
        return loss
    for step in range(5):
        loss = opt2.step(closure2)
        student_f32 = Apex32P(teacher.float(), tie_q=True, tie_g=True)
        student_f32.load_state_dict({k: v.float() for k, v in student.state_dict().items()})
        t_acc, s_acc = eval_model(student_f32)
        print(f"Phase 2 (float64) Step {step+1}: mse={loss.item():.9f}, f32_tok={t_acc:.4f}%, f32_seq={s_acc:.2f}%")
        if s_acc >= 99.95:
            torch.save(student_f32.state_dict(), "distilled_32p_q1_g1.pt")
            for k, v in student_f32.state_dict().items():
                print(f"  {k}: {v.tolist()}")

if __name__ == "__main__":
    main()
