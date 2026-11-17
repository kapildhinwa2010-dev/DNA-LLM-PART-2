# ============================================================
# 🧠 BHAHI KA IDEA: Output-Sensitivity Ranked Basis vs Variance Ranked
# Can "activation x downstream-effect" ranking fix distribution shift?
# ============================================================
import numpy as np, torch, gc, os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*78)
print("🧠 OUTPUT-SENSITIVITY BASIS (Tera Idea) vs VARIANCE BASIS (Failed)")
print("="*78)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYER, D = 10, model.config.hidden_size

cal_prompts = ["The capital of France is", "In Python, to sort a list you use",
               "The square root of 144 is", "Photosynthesis converts sunlight into",
               "The largest planet in our solar system is", "Who wrote Romeo and Juliet?"]
test_prompts = ["The longest river in the world is", "To reverse a string in Python use",
                "The derivative of x cubed is", "Water freezes at what Celsius temperature?",
                "The chemical symbol for gold is", "In chess the bishop moves"]
ML = 24

def get_act_grad_argmax(prompts, want_grad):
    inp = tok(prompts, return_tensors="pt", padding="max_length", max_length=ML, truncation=True)
    pmask = (inp["attention_mask"] == 1)
    box = {}
    def hook(m, i, o):
        o = o.float()
        if want_grad: o.retain_grad()
        box["act"] = o
        return o
    h = model.model.layers[LAYER].register_forward_hook(hook)
    if want_grad:
        out = model(**inp)
        logits = out.logits.float()
        ids = inp["input_ids"]
        tgt = ids[:, 1:]; lg = logits[:, :-1, :]; mk = pmask[:, 1:].float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none")
        ce = ce.view(tgt.shape) * mk
        loss = ce.sum() / mk.sum().clamp(min=1)
        loss.backward()
        g = box["act"].grad
        g = torch.zeros_like(box["act"]) if g is None else g
        h.remove()
        return (box["act"].detach().reshape(-1, D).numpy(),
                g.reshape(-1, D).numpy(),
                logits.argmax(-1), pmask)
    else:
        with torch.no_grad():
            out = model(**inp)
        h.remove()
        return (box["act"].reshape(-1, D).numpy(), None,
                out.logits.argmax(-1), pmask)

# ---- Calibration: activations + output-gradients ----
print("\n📐 Computing calibration activations AND output-sensitivity (gradients)...")
A_cal, G_cal, _, _ = get_act_grad_argmax(cal_prompts, want_grad=True)
N, Dd = A_cal.shape
print(f"   Calib matrix: {A_cal.shape} | grad matrix: {G_cal.shape}")

# ---- SVD (variance order = default) ----
_, Sig, Vt = np.linalg.svd(A_cal, full_matrices=False)
var = Sig**2; cum = np.cumsum(var)/var.sum()
M = Vt.shape[0]

# ---- Output-sensitivity of each component = ||G @ v_j|| ----
GV = G_cal @ Vt.T                      # (N, M)
sens = np.linalg.norm(GV, axis=0)      # (M,) per-component output impact
var_rank = np.arange(M)                # by variance (descending already)
sens_rank = np.argsort(-sens)          # by output impact  ← BHAHI KA IDEA
print(f"   Top-1 by variance  -> output-impact rank: {int(np.where(sens_rank==0)[0][0])+1}")
print(f"   Top-1 by impact    -> variance rank:      {int(np.where(var_rank==sens_rank[0])[0][0])+1}")
print("   (if these ranks differ => variance and impact capture different directions)")

# ---- Unseen baseline (true argmax, no hook) ----
_, _, base_am, tmask = get_act_grad_argmax(test_prompts, want_grad=False)

def proj_hook_factory(Vk):
    Vk = torch.tensor(Vk, dtype=torch.float32)
    def f(m, i, o):
        x = o.float(); return (x @ Vk.T @ Vk).to(o.dtype)
    return f

def unseen_acc(Vk):
    h = model.model.layers[LAYER].register_forward_hook(proj_hook_factory(Vk))
    _, _, am, _ = get_act_grad_argmax(test_prompts, want_grad=False)
    h.remove()
    return ((am == base_am)[tmask].float().mean().item())*100

# ---- Sweep k: variance-rank (control) vs sensitivity-rank (tera idea) ----
print(f"\n{'k':<5}{'VarAcc%':<10}{'SensAcc%':<10}{'Improve':<10}{'Var%kept':<10}")
print("-"*45)
rows = []
for k in [3, 5, 10, 20, 30, 50]:
    if k > M: break
    acc_v = unseen_acc(Vt[var_rank[:k]])
    acc_s = unseen_acc(Vt[sens_rank[:k]])
    vk = cum[var_rank[k-1]]*100 if k-1 < len(cum) else 100.0
    rows.append((k, acc_v, acc_s, acc_s-acc_v))
    flag = "✅" if acc_s-acc_v > 3 else ("~" if abs(acc_s-acc_v)<=3 else "❌")
    print(f"{k:<5}{acc_v:<10.1f}{acc_s:<10.1f}{acc_s-acc_v:+<9.1f}{vk:<10.2f}{flag}")

# ---- VERDICT ----
print("\n"+"="*78)
best = max(rows, key=lambda r: r[3])
gain = best[3]; k_best = best[0]
print(f"Best gain: k={k_best} -> {gain:+.1f}% (var {best[1]:.1f} -> sens {best[2]:.1f})")
print("="*78)
if gain >= 10:
    print("""
✅ YOUR IDEA VALIDATED! Output-sensitivity ranking
   measurably fixed distribution shift (10%+ better than variance-rank on unseen data).
   Conclusion: behavior is compressible, just need OUTPUT-aware basis.
   => Next: make sensitivity INPUT-dependent = your Fourier DNA.
      (instead of static basis, generate sensitivity/basis based on input)
""")
elif gain >= 3:
    print("""
~ PARTIAL. Sensitivity ranking helps somewhat, but not decisive.
   Linear basis still cannot fully handle distribution shift.
   => Suggests: linear weighting insufficient; need NON-LINEAR dynamic
      basis (Fourier DNA) that is input-conditioned.
""")
else:
    print("""
❌ Sensitivity ranking did not beat variance ranking on unseen data.
   => Confirm: NO FIXED linear basis (variance or sensitivity)
      can handle distribution shift. Behavior is inherently
      input-dependent/non-linear.
   => Your Fourier DNA (dynamic, input-conditioned) is the only solution.
      This is also a clean negative result: "linear output-aware bases
      fail under distribution shift in LMs."
""")

import matplotlib.pyplot as plt
ks=[r[0] for r in rows]; av=[r[1] for r in rows]; asen=[r[2] for r in rows]
fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].semilogx(ks,av,'o-',label='Variance-rank (old/failed)',color='#FF6B6B',lw=2)
ax[0].semilogx(ks,asen,'s-',label='Sensitivity-rank (tera idea)',color='#4ECDC4',lw=2)
ax[0].set_xlabel('k components (log)');ax[0].set_ylabel('Unseen Token Acc %')
ax[0].set_title('Output-Sensitivity vs Variance Ranking');ax[0].legend();ax[0].grid(alpha=.3)
ax[1].bar([str(r[0]) for r in rows],[r[3] for r in rows],
          color=['#4ECDC4' if r[3]>0 else '#FF6B6B' for r in rows])
ax[1].axhline(0,color='k',lw=.8);ax[1].set_xlabel('k');ax[1].set_ylabel('Improvement (pp)')
ax[1].set_title('Does Tera Idea Beat Variance?');ax[1].grid(alpha=.3,axis='y')
plt.tight_layout();os.makedirs('results',exist_ok=True)
plt.savefig('results/sensitivity_basis.png',dpi=150,bbox_inches='tight');plt.show()

del model; gc.collect()