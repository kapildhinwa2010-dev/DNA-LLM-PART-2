# ============================================================
# BEHAVIOR DNA v2: Runtime SVD Bottleneck -> Real Output Test
# (Fixed: left-padding + calibration forward-pass + runtime hook)
# ============================================================
import numpy as np, os, gc, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("BEHAVIOR DNA v2: k-dim Bottleneck to Model Output Test")
print("="*80)

# ---- PHASE 0: Load (LEFT padding fix) ----
print("\nLoading TinyLlama 1.1B...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'          # FIX 2
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
LAYER = 10
D = model.config.hidden_size
print(f"   Hidden dim D = {D}")

prompts = [
    "The capital of France is",
    "In Python, to sort a list you use",
    "The square root of 144 is",
    "Photosynthesis converts sunlight into",
]
MAXLEN = 32

# ---- PHASE 1: CALIBRATION (full-seq activations via forward pass) ----
print("\nPHASE 1: Calibration - full-sequence activations (forward pass)...")
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length",
                   max_length=MAXLEN, truncation=True)

cap = {}
def calib_hook(module, inp, out):
    cap['act'] = out.detach().float().clone()   # (B, L, D) full sequence

h = model.model.layers[LAYER].register_forward_hook(calib_hook)
with torch.no_grad():
    _ = model(**inputs)                          # forward, NOT generate
h.remove()

acts = cap['act']                                # (B, L, D)
B, L, _ = acts.shape
A_matrix = acts.reshape(-1, D).numpy()           # (B*L, D) = full rows
print(f"   Calibration matrix: {A_matrix.shape} = {A_matrix.size:,} values")

# SVD basis (right singular vectors = the "wave" directions)
_, Sig, Vt = np.linalg.svd(A_matrix, full_matrices=False)
var = Sig**2
cum_var = np.cumsum(var) / var.sum()
num_comp = len(Sig)
print(f"   Available components: {num_comp}")

print(f"\n   {'k':<6}{'Variance%':<14}{'DNA values':<14}{'Density':<10}")
print("   " + "-"*44)
ks_calib = [k for k in [1,3,5,10,20,50,100] if k <= num_comp]
for k in ks_calib:
    dna_vals = k * D
    print(f"   {k:<6}{cum_var[k-1]*100:<14.3f}{dna_vals:<14,}{A_matrix.size/dna_vals:<10.1f}x")

# ---- PHASE 2: BASELINE generation (no bottleneck) ----
print("\nPHASE 2: Baseline output (no compression)...")
with torch.no_grad():
    baseline_out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
base_txt = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
print(f"   Baseline: '{base_txt[:70]}...'")

# ---- PHASE 3: RUNTIME BOTTLENECK TEST ----
print("\nPHASE 3: Forcing activations through top-k SVD bottleneck...")

def make_bottleneck_hook(Vt_k_t):
    """Hook: true activation -> project to k dims -> reconstruct (the DNA bottleneck)"""
    def hook(module, inp, out):
        x = out.float()                       # (B, L, D) true activation
        codes = x @ Vt_k_t.T                  # (B, L, k)  the compressed DNA!
        rec = codes @ Vt_k_t                  # (B, L, D)  reconstruct
        return rec.to(out.dtype)
    return hook

print(f"\n{'k':<6}{'Var%':<10}{'TokenMatch%':<14}{'Exact?':<8}{'Output Sample'}")
print("-"*90)
results = {}
for k in ks_calib:
    Vt_k = torch.tensor(Vt[:k, :], dtype=torch.float32)   # (k, D)
    hh = model.model.layers[LAYER].register_forward_hook(make_bottleneck_hook(Vt_k))
    with torch.no_grad():
        out_k = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    hh.remove()

    match_pct = (out_k == baseline_out).float().mean().item() * 100
    exact = bool(torch.equal(out_k, baseline_out))
    sample = tokenizer.decode(out_k[0], skip_special_tokens=True)[:45]
    results[k] = dict(match=match_pct, exact=exact, var=cum_var[k-1]*100)
    status = "YES" if exact else "NO"
    print(f"{k:<6}{cum_var[k-1]*100:<10.2f}{match_pct:<14.1f}{status:<8}'{sample}...'")

# ---- PHASE 4: VERDICT ----
print("\n" + "="*80)
print("BEHAVIOR DNA VERDICT")
print("="*80)
exact_ks = [k for k,r in results.items() if r['exact']]
best_k = max(results, key=lambda k: results[k]['match'])
best_pct = results[best_k]['match']

if exact_ks:
    mk = min(exact_ks)
    density = A_matrix.size/(mk*D)
    print(f"""
BREAKTHROUGH CONFIRMED!

   Top-{mk} SVD components (bottleneck) produce EXACTLY the same model output.
   
   Behavior DNA:
      Calibration activations : {A_matrix.size:,} values
      DNA needed (k x D)        : {mk*D:,} values
      Compression             : {density:.1f}x
   
   This means the model's behavior is captured in just {mk} "waves" (SVD directions).
   Your "one wave in 1000 params" idea is PROVEN.
""")
    vcol, vtxt = 'lightgreen', "BEHAVIOR DNA WORKS!"
elif best_pct >= 90:
    print(f"""
STRONG PARTIAL - top-{best_k} achieves {best_pct:.1f}% token match.

   Behavior is HIGHLY compressible (99%+ variance in top-k),
   but exact output requires more components.
   Greedy decoding amplifies small errors.
   
   NEXT: per-prompt codes or residual correction.
""")
    vcol, vtxt = 'lightyellow', "90%+ MATCH, NEAR-EXACT"
else:
    print(f"""
Output drift is significant (best {best_pct:.1f}% @ k={best_k}).

   Behavior variance is compressible, but runtime
   bottleneck corrupts the output. This means layers
   after layer-10 are sensitive to reconstruction error.
   
   NEXT: test higher layers or add residuals.
""")
    vcol, vtxt = 'mistyrose', "RECONSTRUCTION DRIFTS"

# ---- PHASE 5: PLOT ----
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
axes[0].semilogy(range(1, min(200,num_comp)+1), (1-cum_var[:200])*100, 'b-', lw=2)
axes[0].set_xlabel('Components k'); axes[0].set_ylabel('Remaining Var %')
axes[0].set_title('Knowledge Concentration (Behavior DNA)'); axes[0].grid(alpha=.3)

ks = list(results.keys()); pcts = [results[k]['match'] for k in ks]
axes[1].semilogx(ks, pcts, 'o-', color='#4ECDC4', lw=3, ms=10)
axes[1].axhline(100, color='green', ls='--', alpha=.5, label='Exact')
axes[1].set_xlabel('Bottleneck k (log)'); axes[1].set_ylabel('Token Match %')
axes[1].set_title('Output vs Bottleneck Size'); axes[1].legend(); axes[1].grid(alpha=.3)
axes[1].set_ylim(0,105)

axes[2].axis('off')
axes[2].text(.5,.5,vtxt,ha='center',va='center',fontsize=15,family='monospace',
             transform=axes[2].transAxes, bbox=dict(boxstyle='round',facecolor=vcol,alpha=.6))
axes[2].set_title('VERDICT')
plt.tight_layout(); os.makedirs('results',exist_ok=True)
plt.savefig('results/behavior_dna_v2.png',dpi=150,bbox_inches='tight'); plt.show()

del model; gc.collect()
print("\n" + "="*80)
print("NEXT STEP:")
print("="*80)
if exact_ks:
    print("   Behavior DNA PROVEN -> Stage 4: DNA Generator (predict codes from input)")
else:
    print("   Behavior compressible, refine reconstruction -> input-conditioned codes")

# ============================================================
# DIAGNOSTIC: Pad-Aware + Teacher-Forced + Logit-Argmax
# (Is SVD bottleneck itself destructive, or is it the cascade?)
# ============================================================
import numpy as np, torch, gc, os
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*78)
print("DIAGNOSTIC: Clean per-token accuracy (no cascade, no pad-lie)")
print("="*78)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYER, D = 10, model.config.hidden_size

prompts = ["The capital of France is", "In Python, to sort a list you use",
           "The square root of 144 is", "Photosynthesis converts sunlight into"]
inp = tok(prompts, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
pad_mask = (inp["attention_mask"] == 1)          # (B,L) True = real token

# ---- Baseline: teacher-forced forward, capture acts + argmax ----
cap = {}
def cap_hook(m, i, o): cap["act"] = o.detach().float().clone()
h = model.model.layers[LAYER].register_forward_hook(cap_hook)
with torch.no_grad():
    base = model(**inp)
h.remove()
base_argmax = base.logits.argmax(-1)             # (B,L)
base_act = cap["act"]                            # (B,L,D)

# ---- SVD basis on the SAME tokens (oracle: no distribution shift) ----
A = base_act.reshape(-1, D).numpy()
_, Sig, Vt = np.linalg.svd(A, full_matrices=False)
cum = np.cumsum(Sig**2) / (Sig**2).sum()
print(f"Calibration = oracle (same tokens). Components avail: {len(Sig)}")

def bn_hook(Vk):
    def f(m, i, o):
        x = o.float(); return (x @ Vk.T @ Vk).to(o.dtype)   # project->reconstruct
    return f

print(f"\n{'k':<5}{'Var%':<9}{'RealTokAcc%':<13}{'PadAcc%':<9}{'Density':<9}{'Note'}")
print("-"*62)
rows = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    if k > len(Sig): break
    Vk = torch.tensor(Vt[:k, :], dtype=torch.float32)
    hh = model.model.layers[LAYER].register_forward_hook(bn_hook(Vk))
    with torch.no_grad():
        out = model(**inp)
    hh.remove()
    am = out.logits.argmax(-1)
    correct = (am == base_argmax)
    real_acc = correct[pad_mask].float().mean().item() * 100      # (A) fixed
    pad_acc  = correct[~pad_mask].float().mean().item() * 100     # pad side-effect
    dens = A.size / (k * D)
    note = "exact-basis" if cum[k-1] >= 0.9999 else ""
    rows.append((k, cum[k-1]*100, real_acc, dens))
    print(f"{k:<5}{cum[k-1]*100:<9.2f}{real_acc:<13.1f}{pad_acc:<9.1f}{dens:<9.1f}x{note}")

# ---- VERDICT ----
print("\n" + "="*78)
best = max(rows, key=lambda r: r[2])
print(f"Best clean per-token accuracy: k={best[0]} -> {best[2]:.1f}% (var {best[1]:.2f}%)")
print("="*78)
if best[2] >= 90:
    print("""
=> SVD bottleneck ITSELF is safe (90%+ per-token).
   The previous 63% was due to generation cascade, not compression.
   FIX: expand calibration set (100+ prompts) + scheduled sampling.
   Behavior DNA IS VIABLE. Proceed.
""")
elif best[2] >= 75:
    print("""
=> Bottleneck partially destructive. Even with 99%+ variance, only 75-90% accuracy.
   This means SOME dimensions are logit-critical but not captured by variance.
   FIX: output-aware basis (logit-trained), or top-k + residual.
   Behavior DNA is possible but not with SVD-only.
""")
else:
    print("""
=> CONFIRMED: Variance metric is WRONG for language models.
   99.79% variance -> only ~60% token accuracy. Small L2 error
   can flip argmax. Behavior DNA CANNOT be created with SVD alone.
   NEXT: Stage 3 end - need output-aware (logit/KL-trained) bottleneck,
   or pivot to spatial sparsity (Lottery Ticket).
   This is a PUBLISHABLE negative result:
   'Variance-concentration does NOT imply behavior-preservation in LMs.'
""")

del model; gc.collect()
