 ==============================================================================
🧪 MISSION 1.14: SIGNATURE + CORRECTION + TOP-K + HYBRID DNA SWEEP
==============================================================================
Loading weights: 100% 290/290 [00:01<00:00, 283.76it/s]✅ Model loaded | layers=24 | hidden=896 | vocab=151936

📐 Computing DNA bases and sensitive dimensions...
✅ Bases + sensitive dims ready (4.4s)

🧠 Precomputing calibration trajectories...
✅ Calibration trajectories ready: 4 prompts

🧠 Precomputing unseen teacher-forced trajectories...
✅ Unseen trajectories ready: 2 prompts x 3 steps

==============================================================================
🧬 CONFIG: A plain+hidden+target k1 (old control)
==============================================================================

Prompt: 'The capital of Japan is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(28.7%) __(14.0%)Paris(30.4%) ______(12.8%)        🔴 FLIP  0.999
1    'located'     in(45.7%) on(32.0%)man(10.1%) (n(3.0%)               🔴 FLIP  0.988
2    'in'          the(44.3%) which(26.4%)the(29.6%) (26.4%)                🟢 YES   0.992

Prompt: 'The capital of Mexico is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(25.3%) the(9.6%)Paris(30.4%) ______(12.8%)        🔴 FLIP  0.998
1    'located'     in(56.4%) on(21.6%)man(10.1%) (n(3.0%)               🔴 FLIP  0.989
2    'in'          the(45.2%) which(32.3%)the(29.2%) (25.8%)                🟢 YES   0.992

📊 A plain+hidden+target k1 (old control) → Match Rate: 33.3%

==============================================================================
🧬 CONFIG: B plain+signature+correction top5
==============================================================================

Prompt: 'The capital of Japan is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(28.7%) __(14.0%)______(18.1%) ____(12.6%)         🔴 FLIP  0.999
1    'located'     in(45.7%) on(32.0%)______(8.2%) the(8.1%)            🔴 FLIP  0.988
2    'in'          the(44.3%) which(26.4%)the(27.1%) a(14.5%)               🟢 YES   0.992

Prompt: 'The capital of Mexico is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(25.3%) the(9.6%)______(17.7%) ____(12.9%)         🔴 FLIP  0.998
1    'located'     in(56.4%) on(21.6%)(8.4%) the(7.9%)                  🔴 FLIP  0.989
2    'in'          the(45.2%) which(32.3%)the(26.4%) (15.2%)                🟢 YES   0.992

📊 B plain+signature+correction top5 → Match Rate: 33.3%

==============================================================================
🧬 CONFIG: C hybrid+signature+correction top5
==============================================================================

Prompt: 'The capital of Japan is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(28.7%) __(14.0%)______(19.4%) ____(12.3%)         🔴 FLIP  0.998
1    'located'     in(45.7%) on(32.0%)______(9.1%) ((7.2%)              🔴 FLIP  0.983
2    'in'          the(44.3%) which(26.4%)the(34.3%) a(13.1%)               🟢 YES   0.985

Prompt: 'The capital of Mexico is'
Step Context       Original Top-2                    Rescued Top-2                     Match   Sim
--------------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(25.3%) the(9.6%)______(18.5%) ____(12.6%)         🔴 FLIP  0.996
1    'located'     in(56.4%) on(21.6%)((7.8%) ______(7.7%)              🔴 FLIP  0.985
2    'in'          the(45.2%) which(32.3%)the(33.6%) a(13.2%)               🟢 YES   0.986

📊 C hybrid+signature+correction top5 → Match Rate: 33.3%

==============================================================================
🏆 FINAL SWEEP VERDICT
==============================================================================
 33.3%  A plain+hidden+target k1 (old control)
 33.3%  B plain+signature+correction top5
 33.3%  C hybrid+signature+correction top5

Interpretation:
❌ Abhi bhi major drift hai.
   Matlab sirf final hidden rescue kaafi nahi; error layers ke andar non-linearly entangle ho chuka hai.
   Next real fix: DNA-aware distillation / LoRA fine-tuning, ya layer-wise concept correction.

⏱️ Total time: 35.4s
64# ============================================================
# 🧱 MISSION 1.15: CEMENT FLOW MAP
# Ant-bil analogy: andar colored cement bharo, layer-wise flow map banao
# Original vs DNA flow collision detection
# ============================================================
import os, torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# CPU threads
try:
    torch.set_num_threads(max(1, os.cpu_count() - 1))
except Exception:
    pass

t0 = time.time()
print("=" * 78)
print("🧱 MISSION 1.15: CEMENT FLOW MAP — LAYER-WISE KNOWLEDGE ROUTING")
print("=" * 78)

# ------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------
RANK = 16          # DNA compression rank
STEPS = 2          # teacher-forced generation steps; set 1 if slow
THRESH = 0.05      # drift threshold for "first bad layer"

# ------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float32,
    low_cpu_mem_usage=True
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

print(f"✅ Model loaded | layers={NL} | hidden={D}")

# ------------------------------------------------------------
# 2. Calibration prompts -> DNA bases
# ------------------------------------------------------------
cal_prompts = [
    "The capital of France is",
    "The president of USA in 2009 was",
    "The chemical symbol for gold is",
    "In Python, to sort a list you use",
]

cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

print("📐 Capturing calibration deltas for DNA basis...")
boxes = {l: {} for l in range(NL)}

def make_capture_hook(l_idx):
    def hook(m, i, o):
        boxes[l_idx]["in"] = i[0].detach().float()
        boxes[l_idx]["out"] = o.detach().float()
        return o
    return hook

capture_hooks = [
    model.model.layers[l].register_forward_hook(make_capture_hook(l))
    for l in range(NL)
]

with torch.no_grad():
    model(**cal_inp)

for h in capture_hooks:
    h.remove()

print("⏳ Computing Rank-16 DNA bases...")
bases = {}
for l in range(NL):
    delta = boxes[l]["out"] - boxes[l]["in"]
    flat = delta.reshape(-1, D)[cal_mask.reshape(-1)].numpy()
    mu = flat.mean(0)
    centered = flat - mu

    # SVD basis
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    Vk = Vt[:RANK]

    bases[l] = (
        torch.tensor(Vk, dtype=torch.float32),
        torch.tensor(mu, dtype=torch.float32)
    )

print(f"✅ DNA bases ready ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 3. Permanent DNA hooks, controlled by STATE
# ------------------------------------------------------------
STATE = {"dna": False}

def make_dna_hook(l_idx):
    def hook(m, i, o):
        if not STATE["dna"]:
            return o

        x = i[0].float()
        delta = o.float() - x
        fd = delta.reshape(-1, D)

        Vk, mu = bases[l_idx]
        proj = ((fd - mu) @ Vk.T) @ Vk + mu

        return (x + proj.reshape(o.shape)).to(o.dtype)
    return hook

dna_hooks = [
    model.model.layers[l].register_forward_hook(make_dna_hook(l))
    for l in range(NL)
]

# ------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------
def get_flow(ids, dna=False):
    """
    Returns:
      H: (NL, D) last-token hidden state for every layer
      top_idx: (5,) top token ids from final logits
      top_prob: (5,) probabilities
    """
    STATE["dna"] = dna
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states[0] = embeddings, hidden_states[l+1] = layer l output
    H = torch.stack(
        [out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)],
        dim=0
    )

    logits = out.logits[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    tk = torch.topk(probs, 5)

    return H, tk.indices, tk.values


def teacher_sequences(prompt, steps):
    """
    Generate teacher-forced sequences using ORIGINAL model only.
    Returns list of input_ids:
      [prompt, prompt+tok1, prompt+tok1+tok2, ...]
    """
    ids = tok(prompt, return_tensors="pt").input_ids
    seqs = [ids.clone()]

    STATE["dna"] = False
    with torch.no_grad():
        for _ in range(steps):
            out = model(ids)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
            seqs.append(ids.clone())

    return seqs


def decode_top(idx_tensor, n=1):
    return [tok.decode([int(t)]).strip() for t in idx_tensor[:n]]


# ------------------------------------------------------------
# 5. Unseen prompts = colored cement probes
# ------------------------------------------------------------
eval_prompts = [
    "The capital of Japan is",
    "The capital of Mexico is",
    "The chemical symbol for silver is",
]

eval_labels = ["Japan", "Mexico", "Silver"]

records = []

print("\n" + "=" * 78)
print("🧪 RUNNING CEMENT PROBES: Original vs DNA Flow")
print("=" * 78)

print(
    f"{'Prompt':<9}{'Step':<5}{'FirstDrift':<11}{'MaxDrift':<9}"
    f"{'L0':<6}{'L5':<6}{'L11':<6}{'L23':<6}{'Top3':<6}{'Orig -> DNA'}"
)
print("-" * 95)

for pi, (prompt, label) in enumerate(zip(eval_prompts, eval_labels)):
    seqs = teacher_sequences(prompt, STEPS)

    for st, ids in enumerate(seqs):
        H_o, top_o, prob_o = get_flow(ids, dna=False)
        H_d, top_d, prob_d = get_flow(ids, dna=True)

        # Layer-wise directional drift
        cos = F.cosine_similarity(H_o, H_d, dim=1)
        drift = (1.0 - cos).clamp(min=0.0)

        bad = drift > THRESH
        first_bad = int(torch.argmax(bad.to(torch.int64)).item()) if bad.any() else -1
        max_bad = int(torch.argmax(drift).item())

        # Top-3 output overlap
        set_o = set(top_o[:3].tolist())
        set_d = set(top_d[:3].tolist())
        overlap = len(set_o & set_d) / 3.0

        o1 = decode_top(top_o, 1)[0]
        d1 = decode_top(top_d, 1)[0]

        first_s = "-" if first_bad < 0 else f"L{first_bad}"
        max_s = f"L{max_bad}"

        print(
            f"{label:<9}{st:<5}{first_s:<11}{max_s:<9}"
            f"{drift[0]:<6.3f}{drift[5]:<6.3f}{drift[11]:<6.3f}{drift[23]:<6.3f}"
            f"{overlap*100:<5.0f}% {o1} -> {d1}"
        )

        records.append({
            "pi": pi,
            "label": label,
            "step": st,
            "H_o": H_o,
            "H_d": H_d,
            "top_o": top_o,
            "top_d": top_d,
            "prob_o": prob_o,
            "prob_d": prob_d,
            "drift": drift,
            "first_bad": first_bad,
            "max_bad": max_bad,
            "overlap": overlap,
        })

# ------------------------------------------------------------
# 6. Sample cement signature: Layer 11 top active dimensions
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("🧱 SAMPLE CEMENT SIGNATURE — Layer 11 Top-8 Active Dims (Step 0)")
print("=" * 78)

for rec in records:
    if rec["step"] != 0:
        continue

    l = 11
    dims_o = torch.topk(rec["H_o"][l].abs(), 8).indices.tolist()
    dims_d = torch.topk(rec["H_d"][l].abs(), 8).indices.tolist()

    print(f"{rec['label']:<9} ORIG dims: {dims_o}")
    print(f"{rec['label']:<9} DNA  dims: {dims_d}")
    print()

# ------------------------------------------------------------
# 7. Identity collapse check:
# Does DNA make Japan/Mexico/Silver too similar?
# ------------------------------------------------------------
print("=" * 78)
print("🧩 IDENTITY COLLAPSE CHECK — Pairwise Flow Distance")
print("=" * 78)

print(f"{'Step':<5}{'Pair':<18}{'OrigDist':<10}{'DNADist':<10}{'Ratio':<8}{'Verdict'}")
print("-" * 70)

collapse_ratios = []

for st in range(STEPS + 1):
    step_recs = [r for r in records if r["step"] == st]

    vecs_o = []
    vecs_d = []
    names = []

    for r in step_recs:
        # Concatenate all layer hidden states into one flow vector
        vo = F.normalize(r["H_o"].reshape(-1), dim=0)
        vd = F.normalize(r["H_d"].reshape(-1), dim=0)
        vecs_o.append(vo)
        vecs_d.append(vd)
        names.append(r["label"])

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            orig_dist = 1.0 - torch.dot(vecs_o[i], vecs_o[j]).item()
            dna_dist = 1.0 - torch.dot(vecs_d[i], vecs_d[j]).item()

            if orig_dist > 1e-4:
                ratio = dna_dist / orig_dist
            else:
                ratio = 1.0

            collapse_ratios.append(ratio)

            pair = f"{names[i]}-{names[j]}"

            if orig_dist < 0.02:
                verdict = "⚪ TOO SIMILAR"
            elif ratio < 0.50:
                verdict = "🔴 COLLAPSED"
            elif ratio < 0.80:
                verdict = "🟡 PARTIAL"
            else:
                verdict = "🟢 OK"

            print(
                f"{st:<5}{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}"
                f"{ratio:<8.3f}{verdict}"
            )

# ------------------------------------------------------------
# 8. Save physical 3D flow map artifact
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)
map_file = "results/mission_1_15_cement_flow_map.npz"

save_dict = {
    "H_o": np.stack([r["H_o"].numpy() for r in records]),
    "H_d": np.stack([r["H_d"].numpy() for r in records]),
    "drift": np.stack([r["drift"].numpy() for r in records]),
    "top_o": np.stack([r["top_o"].numpy() for r in records]),
    "top_d": np.stack([r["top_d"].numpy() for r in records]),
    "prob_o": np.stack([r["prob_o"].numpy() for r in records]),
    "prob_d": np.stack([r["prob_d"].numpy() for r in records]),
    "prompt_idx": np.array([r["pi"] for r in records]),
    "step": np.array([r["step"] for r in records]),
    "first_bad": np.array([r["first_bad"] for r in records]),
    "max_bad": np.array([r["max_bad"] for r in records]),
    "overlap": np.array([r["overlap"] for r in records]),
}

np.savez_compressed(map_file, **save_dict)
map_size_kb = os.path.getsize(map_file) / 1024

print("\n" + "=" * 78)
print(f"💾 3D CEMENT FLOW MAP SAVED: {map_file}")
print(f"📦 Size: {map_size_kb:.2f} KB")
print("=" * 78)

# ------------------------------------------------------------
# 9. Final verdict
# ------------------------------------------------------------
avg_overlap = float(np.mean([r["overlap"] for r in records])) * 100.0

firsts = [r["first_bad"] for r in records if r["first_bad"] >= 0]
median_first = float(np.median(firsts)) if firsts else None

finite_ratios = [x for x in collapse_ratios if np.isfinite(x)]
min_ratio = float(np.min(finite_ratios)) if finite_ratios else 1.0

print("\n🏆 MISSION 1.15 VERDICT")
print("-" * 78)
print(f"Average Top-3 output overlap : {avg_overlap:.1f}%")
print(f"Median first drift layer     : {'None' if median_first is None else f'L{int(median_first)}'}")
print(f"Minimum identity ratio       : {min_ratio:.3f}")
print()

if avg_overlap >= 70.0 and min_ratio >= 0.70:
    print("✅ FLOW MAP RELATIVELY SAFE.")
    print("   DNA ne concept routes kaafi preserve kiye.")
    print("   Next: Physical DNA artifact bana sakte hain.")

elif min_ratio < 0.50:
    print("🔴 IDENTITY COLLAPSE DETECTED.")
    print("   Rank-16 DNA ne alag concepts ko ek jaisa dikha diya.")
    print("   Example: Japan/France/Mexico paths collide ho rahe hain.")
    print("   Next: Routing-aware DNA / sensitive identity subspace preserve karna padega.")

elif median_first is not None and median_first <= 7:
    print("🔴 EARLY-LAYER DRIFT DETECTED.")
    print("   Problem Layer 0-7 mein shuru ho raha hai.")
    print("   Matlab input routing / token identity compression mein kho jaa rahi hai.")
    print("   Next: Early layers ko kam compress karo ya identity dims preserve karo.")

elif avg_overlap < 40.0:
    print("⚠️ OUTPUT ROUTING BROKEN.")
    print("   Internal flow maybe similar hai, par final token routing bigad raha hai.")
    print("   Next: LM-head aware correction / concept direction alignment.")

else:
    print("⚠️ PARTIAL DRIFT.")
    print("   Kuch layers ok, kuch drift. Butterfly effect abhi bhi present hai.")
    print("   Next: Layer-wise correction ya DNA-aware distillation.")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")

# cleanup
for h in dna_hooks:
    h.remove()

del model
gc.collect()# ============================================================
# 🧪 MISSION 1.14: ALL POSSIBILITIES SWEEP
# Signature + Correction Blend + Top-K + Hybrid Sensitive Dims
# ============================================================
import torch
import numpy as np
import time
import gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("=" * 78)
print("🧪 MISSION 1.14: SIGNATURE + CORRECTION + TOP-K + HYBRID DNA SWEEP")
print("=" * 78)

# ----------------------------
# 0. Global state
# ----------------------------
COMP_MODE = "off"   # off | plain | hybrid

# ----------------------------
# 1. Model load
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.float32,
    low_cpu_mem_usage=True
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

W_unembed = model.get_output_embeddings().weight.detach().float()  # (vocab, D)
embed_layer = model.model.embed_tokens

print(f"✅ Model loaded | layers={NL} | hidden={D} | vocab={W_unembed.shape[0]}")

# ----------------------------
# 2. Hyperparameters
# ----------------------------
SENS_DIMS = 32       # hybrid me kitne sensitive dims preserve karne hain
RANK = 16            # DNA rank
K_BANK = 16          # bank me per state kitne top target logits store karne hain
K_DNA = 8            # target-rescue me DNA ke top-k suppress karne ke liye
GEN_TOKENS = 2       # calibration trajectory me teacher-generated tokens
STEPS = 3            # unseen prompt par kitne steps track karne hain
TOPK_NEI = 5         # correction blend me kitne neighbors
TEMP = 20.0          # similarity sharpening for top-k weights

cal_prompts = [
    "The capital of France is",
    "The president of USA in 2009 was",
    "The chemical symbol for gold is",
    "In Python, to sort a list you use",
]

unseen_prompts = [
    "The capital of Japan is",
    "The capital of Mexico is",
]

# ----------------------------
# 3. Initial calibration: compute DNA bases + sensitive dims
# ----------------------------
print("\n📐 Computing DNA bases and sensitive dimensions...")

cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

boxes = {l: {} for l in range(NL)}

def make_capture_hook(l_idx):
    def hook(m, i, o):
        boxes[l_idx]["in"] = i[0].detach().float()
        boxes[l_idx]["out"] = o.detach().float()
        return o
    return hook

cap_hooks = [
    model.model.layers[l].register_forward_hook(make_capture_hook(l))
    for l in range(NL)
]

with torch.no_grad():
    model(**cal_inp)

for h in cap_hooks:
    h.remove()

bases = {}
sens = {}

for l in range(NL):
    delta = boxes[l]["out"] - boxes[l]["in"]  # (B, L, D)

    # All non-pad token deltas for SVD basis
    flat = delta.view(-1, D)[cal_mask.view(-1)].numpy()
    mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    bases[l] = (
        torch.tensor(Vt[:RANK], dtype=torch.float32),
        torch.tensor(mu, dtype=torch.float32)
    )

    # Prompt-separable dimensions:
    # dimensions jahan France/Japan/president/gold/python ke average delta alag hain
    prompt_means = []
    for b in range(len(cal_prompts)):
        m = cal_mask[b]
        if m.any():
            prompt_means.append(delta[b][m].mean(0).numpy())
    prompt_means = np.stack(prompt_means, axis=0)  # (P, D)
    var_across_prompts = prompt_means.var(axis=0)  # (D,)
    top_sens = np.argsort(-var_across_prompts)[:SENS_DIMS]
    sens[l] = torch.tensor(top_sens, dtype=torch.long)

print(f"✅ Bases + sensitive dims ready ({time.time()-t0:.1f}s)")

# ----------------------------
# 4. Permanent layer hooks for plain/hybrid DNA
# ----------------------------
def make_layer_hook(l_idx):
    def hook(m, i, o):
        global COMP_MODE
        if COMP_MODE == "off":
            return o

        x = i[0].float()
        delta_true = o.float() - x
        fd = delta_true.view(-1, D)

        Vk, mu = bases[l_idx]

        # Rank-16 compressed delta
        proj = ((fd - mu) @ Vk.T) @ Vk + mu

        # Hybrid: sensitive dims ko original true delta se preserve karo
        if COMP_MODE == "hybrid":
            sd = sens[l_idx]
            proj[:, sd] = fd[:, sd]

        return (x + proj.view(o.shape)).to(o.dtype)
    return hook

perm_hooks = [
    model.model.layers[l].register_forward_hook(make_layer_hook(l))
    for l in range(NL)
]

# ----------------------------
# 5. Helper: mean embedding signature
# ----------------------------
def mean_emb(ids):
    with torch.no_grad():
        e = embed_layer(ids)[0].float()  # (L, D)
    return e.mean(0)

# ----------------------------
# 6. Precompute calibration trajectories from ORIGINAL model
#    (teacher-forced, so bank me original targets/corrections mil sake)
# ----------------------------
print("\n🧠 Precomputing calibration trajectories...")

cal_trajs = []
COMP_MODE = "off"

for p in cal_prompts:
    ids = tok(p, return_tensors="pt").input_ids

    with torch.no_grad():
        # Teacher generate a few continuation tokens
        for _ in range(GEN_TOKENS):
            out = model(ids)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)

        # Original hidden states + logits
        out = model(ids, output_hidden_states=True)

    H_orig = out.hidden_states[-1][0].float()  # (L, D)
    logits_orig = out.logits[0].float()        # (L, vocab)
    L = ids.shape[1]

    top_idx_list = []
    top_val_list = []
    for i in range(L - 1):
        tk = torch.topk(logits_orig[i], K_BANK)
        top_idx_list.append(tk.indices)
        top_val_list.append(tk.values)

    top_idx = torch.stack(top_idx_list, dim=0)  # (L-1, K_BANK)
    top_val = torch.stack(top_val_list, dim=0)  # (L-1, K_BANK)

    # Prefix mean embedding signature for each position
    emb = embed_layer(ids)[0].float()           # (L, D)
    cum = emb.cumsum(0)
    counts = torch.arange(1, L + 1, dtype=torch.float32).unsqueeze(1)
    sig = (cum / counts)[:L - 1]                # (L-1, D)

    cal_trajs.append({
        "prompt": p,
        "ids": ids,
        "orig_H": H_orig,
        "top_idx": top_idx,
        "top_val": top_val,
        "sig": sig,
    })

print(f"✅ Calibration trajectories ready: {len(cal_trajs)} prompts")

# ----------------------------
# 7. Precompute unseen trajectories + original ground truth
# ----------------------------
print("\n🧠 Precomputing unseen teacher-forced trajectories...")

unseen_trajs = []

for p in unseen_prompts:
    ids = tok(p, return_tensors="pt").input_ids
    prefixes = []
    orig_top2 = []

    with torch.no_grad():
        for step in range(STEPS):
            out = model(ids)
            probs = F.softmax(out.logits[0, -1].float(), dim=-1)
            tk = torch.topk(probs, 2)

            prefixes.append(ids.clone())
            orig_top2.append((tk.indices.clone(), tk.values.clone()))

            nxt = tk.indices[0].unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=-1)

    unseen_trajs.append({
        "prompt": p,
        "prefixes": prefixes,
        "orig_top2": orig_top2,
    })

print(f"✅ Unseen trajectories ready: {len(unseen_trajs)} prompts x {STEPS} steps")

# ----------------------------
# 8. Build concept bank for a given compression + key mode
# ----------------------------
def build_bank(comp_mode, key_mode):
    global COMP_MODE
    COMP_MODE = comp_mode

    keys = []
    corrs = []
    tidx = []
    tval = []

    with torch.no_grad():
        for tr in cal_trajs:
            out = model(tr["ids"], output_hidden_states=True)
            H_cmp = out.hidden_states[-1][0].float()  # (L, D)
            L = tr["ids"].shape[1]

            for i in range(L - 1):
                h = H_cmp[i]
                corr = tr["orig_H"][i] - h

                if key_mode == "hidden":
                    key = h
                else:
                    # signature key: hidden + prefix embedding mean
                    key = torch.cat([h, tr["sig"][i]])

                key = F.normalize(key, dim=0)

                keys.append(key)
                corrs.append(corr)
                tidx.append(tr["top_idx"][i])
                tval.append(tr["top_val"][i])

    COMP_MODE = "off"

    return {
        "keys": torch.stack(keys, dim=0),     # (N, D or 2D)
        "corrs": torch.stack(corrs, dim=0),   # (N, D)
        "tidx": torch.stack(tidx, dim=0),     # (N, K_BANK)
        "tval": torch.stack(tval, dim=0),     # (N, K_BANK)
    }

# ----------------------------
# 9. Rescue one step from bank
# ----------------------------
def rescue_from_bank(bank, key_mode, rescue_mode, topk_nei, prefix_ids, comp_mode):
    global COMP_MODE
    COMP_MODE = comp_mode

    with torch.no_grad():
        out = model(prefix_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float()       # (D,)
        logits_cmp = out.logits[0, -1].float()         # (vocab,)

        if key_mode == "hidden":
            key = h
        else:
            sig = mean_emb(prefix_ids)
            key = torch.cat([h, sig])

        key = F.normalize(key, dim=0)
        sims = bank["keys"] @ key

        k = min(topk_nei, sims.numel())
        topv, topi = torch.topk(sims, k)

        if rescue_mode == "target":
            # Old Mission 1.12/1.13 style: retrieve target logits and solve min-norm correction
            j = topi[0]
            ret_idx = bank["tidx"][j]
            ret_val = bank["tval"][j]

            dna_top = torch.topk(logits_cmp, K_DNA)
            union = torch.unique(torch.cat([ret_idx, dna_top.indices]))

            min_ret = float(ret_val.min().item())
            target = torch.full((union.numel(),), min_ret - 10.0, dtype=torch.float32)

            for ri, rv in zip(ret_idx.tolist(), ret_val.tolist()):
                pos = torch.where(union == ri)[0]
                if pos.numel() > 0:
                    target[pos[0]] = float(rv)

            W_top = W_unembed[union]  # (k, D)
            WWT = W_top @ W_top.T
            WWT = WWT + 1e-3 * torch.eye(WWT.shape[0], dtype=WWT.dtype)
            pinv = torch.linalg.pinv(WWT)

            cur = h @ W_top.T
            corr = W_top.T @ (pinv @ (target - cur))
            h_res = h + corr

        else:
            # correction blend: directly add remembered hidden-state correction
            w = torch.softmax((topv - topv.max()) * TEMP, dim=0)
            corr = (bank["corrs"][topi] * w.unsqueeze(1)).sum(0)
            h_res = h + corr

        logits_res = h_res @ W_unembed.T
        probs_res = F.softmax(logits_res, dim=-1)
        top2_res = torch.topk(probs_res, 2)

    COMP_MODE = "off"
    return top2_res, float(topv[0].item())

# ----------------------------
# 10. Evaluate configs
# ----------------------------
configs = [
    ("A plain+hidden+target k1 (old control)", "plain", "hidden", "target", 1),
    ("B plain+signature+correction top5",      "plain", "signature", "correction", TOPK_NEI),
    ("C hybrid+signature+correction top5",     "hybrid", "signature", "correction", TOPK_NEI),
]

summary = []

for name, comp_mode, key_mode, rescue_mode, topk_nei in configs:
    print("\n" + "=" * 78)
    print(f"🧬 CONFIG: {name}")
    print("=" * 78)

    bank = build_bank(comp_mode, key_mode)

    all_matches = []

    for tr in unseen_trajs:
        print(f"\nPrompt: '{tr['prompt']}'")
        print(f"{'Step':<5}{'Context':<14}{'Original Top-2':<34}{'Rescued Top-2':<34}{'Match':<8}{'Sim'}")
        print("-" * 110)

        for step in range(STEPS):
            prefix_ids = tr["prefixes"][step]
            orig_idx, orig_prob = tr["orig_top2"][step]

            top2_res, sim = rescue_from_bank(
                bank=bank,
                key_mode=key_mode,
                rescue_mode=rescue_mode,
                topk_nei=topk_nei,
                prefix_ids=prefix_ids,
                comp_mode=comp_mode,
            )

            w_orig = [tok.decode([int(t)]).strip() for t in orig_idx]
            p_orig = orig_prob * 100

            w_res = [tok.decode([int(t)]).strip() for t in top2_res.indices]
            p_res = top2_res.values * 100

            match_int = int(orig_idx[0] == top2_res.indices[0])
            all_matches.append(match_int)
            match = "🟢 YES" if match_int else "🔴 FLIP"

            ctx = "[PROMPT]" if step == 0 else repr(tok.decode([int(prefix_ids[0, -1])]).strip())

            print(
                f"{step:<5}{ctx:<14}"
                f"{w_orig[0]}({p_orig[0]:.1f}%) {w_orig[1]}({p_orig[1]:.1f}%)".ljust(34)
                +
                f"{w_res[0]}({p_res[0]:.1f}%) {w_res[1]}({p_res[1]:.1f}%)".ljust(34)
                +
                f"{match:<8}{sim:.3f}"
            )

    acc = float(np.mean(all_matches) * 100)
    summary.append((name, acc))
    print(f"\n📊 {name} → Match Rate: {acc:.1f}%")

# ----------------------------
# 11. Final verdict
# ----------------------------
print("\n" + "=" * 78)
print("🏆 FINAL SWEEP VERDICT")
print("=" * 78)
for name, acc in summary:
    print(f"{acc:5.1f}%  {name}")

best_name, best_acc = max(summary, key=lambda x: x[1])
old_acc = summary[0][1]

print("\nInterpretation:")
if best_acc >= 80:
    print("✅ BAHUT BADHIYA! Signature + correction + hybrid ne drift ko kaafi halka kar diya.")
    print("   Next: isko standalone banane ke liye learned concept predictor / small corrector train karo.")
elif best_acc > old_acc + 15:
    print("⚠️ IMPROVEMENT MILA, par abhi full usable nahi.")
    print("   Tera direction sahi hai: concept dimensions + memory correction help kar rahi hai.")
    print("   Next: bank size badhao, top-k blend tune karo, ya sensitive dims ko stored DNA me convert karo.")
else:
    print("❌ Abhi bhi major drift hai.")
    print("   Matlab sirf final hidden rescue kaafi nahi; error layers ke andar non-linearly entangle ho chuka hai.")
    print("   Next real fix: DNA-aware distillation / LoRA fine-tuning, ya layer-wise concept correction.")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")

# cleanup
for h in perm_hooks:
    h.remove()
del model
gc.collect() ==============================================================================
🧱 MISSION 1.15: CEMENT FLOW MAP — LAYER-WISE KNOWLEDGE ROUTING
==============================================================================
Loading weights: 100% 290/290 [00:01<00:00, 251.59it/s]✅ Model loaded | layers=24 | hidden=896
📐 Capturing calibration deltas for DNA basis...
⏳ Computing Rank-16 DNA bases...
✅ DNA bases ready (4.5s)

==============================================================================
🧪 RUNNING CEMENT PROBES: Original vs DNA Flow
==============================================================================
Prompt   Step FirstDrift MaxDrift L0    L5    L11   L23   Top3  Orig -> DNA
-----------------------------------------------------------------------------------------------
Japan    0    L3         L23      0.029 0.085 0.087 0.142 33   % located -> ____
Japan    1    L0         L0       0.571 0.378 0.355 0.289 0    % in -> __
Japan    2    L0         L5       0.141 0.295 0.270 0.175 33   % the -> the
Mexico   0    L3         L23      0.032 0.077 0.075 0.137 0    % located -> ____
Mexico   1    L0         L0       0.569 0.371 0.355 0.235 0    % in -> __
Mexico   2    L0         L5       0.140 0.297 0.264 0.148 33   % the -> a
Silver   0    L2         L23      0.033 0.072 0.074 0.090 100  % ____ -> ____
Silver   1    L0         L0       0.635 0.532 0.422 0.277 0    % . -> ____
Silver   2    L0         L23      0.465 0.418 0.451 0.997 0    % A -> __

==============================================================================
🧱 SAMPLE CEMENT SIGNATURE — Layer 11 Top-8 Active Dims (Step 0)
==============================================================================
Japan     ORIG dims: [490, 386, 570, 208, 646, 817, 67, 142]
Japan     DNA  dims: [490, 570, 386, 208, 646, 67, 757, 817]

Mexico    ORIG dims: [490, 570, 386, 208, 646, 817, 142, 595]
Mexico    DNA  dims: [490, 570, 386, 208, 646, 67, 757, 817]

Silver    ORIG dims: [490, 450, 386, 208, 570, 559, 142, 396]
Silver    DNA  dims: [490, 208, 570, 386, 450, 559, 67, 142]

==============================================================================
🧩 IDENTITY COLLAPSE CHECK — Pairwise Flow Distance
==============================================================================
Step Pair              OrigDist  DNADist   Ratio   Verdict
----------------------------------------------------------------------
0    Japan-Mexico      0.1119    0.0001    0.001   🔴 COLLAPSED
0    Japan-Silver      0.3111    0.0632    0.203   🔴 COLLAPSED
0    Mexico-Silver     0.3467    0.0625    0.180   🔴 COLLAPSED
1    Japan-Mexico      0.0234    0.0001    0.003   🔴 COLLAPSED
1    Japan-Silver      0.4391    0.0548    0.125   🔴 COLLAPSED
1    Mexico-Silver     0.4166    0.0545    0.131   🔴 COLLAPSED
2    Japan-Mexico      0.0761    0.0000    0.000   🔴 COLLAPSED
2    Japan-Silver      0.8242    0.0364    0.044   🔴 COLLAPSED
2    Mexico-Silver     0.8685    0.0364    0.042   🔴 COLLAPSED

==============================================================================
💾 3D CEMENT FLOW MAP SAVED: results/mission_1_15_cement_flow_map.npz
📦 Size: 1413.22 KB
==============================================================================

🏆 MISSION 1.15 VERDICT
------------------------------------------------------------------------------
Average Top-3 output overlap : 22.2%
Median first drift layer     : L0
Minimum identity ratio       : 0.000

🔴 IDENTITY COLLAPSE DETECTED.
   Rank-16 DNA ne alag concepts ko ek jaisa dikha diya.
   Example: Japan/France/Mexico paths collide ho rahe hain.
   Next: Routing-aware DNA / sensitive identity subspace preserve karna padega.

⏱️ Total time: 18.5s
64# ============================================================
# 🧱 MISSION 1.16: SELECTIVE DEPTH DNA SWEEP
# Ant-bil idea: kuch tunnels ko cement se mat bharo,
# identity-critical early layers ko full rakho.
# ============================================================
import os
import torch
import numpy as np
import time
import gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# CPU threads
try:
    cpu = os.cpu_count() or 1
    torch.set_num_threads(max(1, cpu - 1))
except Exception:
    pass

t0 = time.time()
print("=" * 78)
print("🧱 MISSION 1.16: SELECTIVE DEPTH DNA SWEEP")
print("=" * 78)

# ------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------
RANK = 16          # DNA rank for compressed layers
STEPS = 2          # teacher-forced steps: prompt, +1 token, +2 tokens
THRESH = 0.05      # drift threshold for first bad layer
FOCUS_CF = 8       # detailed report for this config

# ------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
except TypeError:
    # Older transformers fallback
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

tok = AutoTokenizer.from_pretrained(model_name)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# configs after NL known
CONFIGS = [0, 4, 8, 12, NL]

print(f"✅ Model loaded | layers={NL} | hidden={D} | rank={RANK}")
print(f"⚙️ Configs: keep-first-N-layers-full = {CONFIGS}")

# ------------------------------------------------------------
# 2. Calibration prompts -> DNA bases
# ------------------------------------------------------------
cal_prompts = [
    "The capital of France is",
    "The president of USA in 2009 was",
    "The chemical symbol for gold is",
    "In Python, to sort a list you use",
]

cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

print("\n📐 Capturing calibration deltas...")
boxes = {l: {} for l in range(NL)}

def make_capture_hook(l_idx):
    def hook(m, i, o):
        boxes[l_idx]["in"] = i[0].detach().float()
        boxes[l_idx]["out"] = o.detach().float()
        return o
    return hook

capture_hooks = [
    model.model.layers[l].register_forward_hook(make_capture_hook(l))
    for l in range(NL)
]

with torch.no_grad():
    model(**cal_inp)

for h in capture_hooks:
    h.remove()

print("⏳ Computing Rank-16 DNA bases...")
bases = {}
for l in range(NL):
    delta = boxes[l]["out"] - boxes[l]["in"]
    flat = delta.reshape(-1, D)[cal_mask.reshape(-1)].numpy()

    if flat.shape[0] == 0:
        # extremely safe fallback
        bases[l] = (
            torch.zeros((1, D), dtype=torch.float32),
            torch.zeros(D, dtype=torch.float32)
        )
        continue

    mu = flat.mean(0)
    centered = flat - mu

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    k = min(RANK, Vt.shape[0])

    bases[l] = (
        torch.tensor(Vt[:k], dtype=torch.float32),
        torch.tensor(mu, dtype=torch.float32)
    )

print(f"✅ DNA bases ready ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 3. Permanent DNA hooks, controlled by STATE
# ------------------------------------------------------------
STATE = {
    "dna": False,
    "compress_from": 0
}

def make_dna_hook(l_idx):
    def hook(m, i, o):
        if not STATE["dna"]:
            return o

        # Keep first `compress_from` layers uncompressed
        if l_idx < STATE["compress_from"]:
            return o

        x = i[0].float()
        delta = o.float() - x
        fd = delta.reshape(-1, D)

        Vk, mu = bases[l_idx]
        proj = ((fd - mu) @ Vk.T) @ Vk + mu

        return (x + proj.reshape(o.shape)).to(o.dtype)
    return hook

dna_hooks = [
    model.model.layers[l].register_forward_hook(make_dna_hook(l))
    for l in range(NL)
]

# ------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------
def get_flow(ids, dna=False, compress_from=0):
    """
    Returns:
      H: (NL, D) last-token hidden state for every layer
      top_idx: (5,) top token ids
      top_prob: (5,) probabilities
    """
    STATE["dna"] = dna
    STATE["compress_from"] = compress_from

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states[0] = embeddings
    # hidden_states[l+1] = output of layer l
    H = torch.stack(
        [out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)],
        dim=0
    )

    logits = out.logits[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    tk = torch.topk(probs, 5)

    STATE["dna"] = False
    return H, tk.indices, tk.values


def teacher_sequences(prompt, steps):
    """
    Generate teacher-forced sequences using ORIGINAL model only.
    Returns list of input_ids:
      [prompt, prompt+tok1, prompt+tok1+tok2, ...]
    """
    ids = tok(prompt, return_tensors="pt").input_ids
    seqs = [ids.clone()]

    old = STATE["dna"]
    STATE["dna"] = False

    with torch.no_grad():
        for _ in range(steps):
            out = model(ids)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
            seqs.append(ids.clone())

    STATE["dna"] = old
    return seqs


def decode_top(idx_tensor, n=1):
    out = []
    for t in idx_tensor[:n]:
        s = tok.decode([int(t)]).strip()
        out.append(s if s else "<empty>")
    return out


# ------------------------------------------------------------
# 5. Unseen prompts = colored cement probes
# ------------------------------------------------------------
eval_prompts = [
    "The capital of Japan is",
    "The capital of Mexico is",
    "The chemical symbol for silver is",
]
eval_labels = ["Japan", "Mexico", "Silver"]

print("\n🧠 Precomputing teacher-forced unseen sequences...")
unseen_seqs = []
for p in eval_prompts:
    unseen_seqs.append(teacher_sequences(p, STEPS))

print("🧠 Precomputing original-model flows...")
orig_flows = []
for seqs in unseen_seqs:
    step_flows = []
    for ids in seqs:
        H_o, top_o, prob_o = get_flow(ids, dna=False, compress_from=0)
        step_flows.append((H_o, top_o, prob_o))
    orig_flows.append(step_flows)

print(f"✅ Original flows ready ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 6. Evaluate one selective-depth config
# ------------------------------------------------------------
def evaluate_config(compress_from):
    records = []

    for pi, (prompt, label) in enumerate(zip(eval_prompts, eval_labels)):
        seqs = unseen_seqs[pi]

        for st, ids in enumerate(seqs):
            H_o, top_o, prob_o = orig_flows[pi][st]
            H_d, top_d, prob_d = get_flow(ids, dna=True, compress_from=compress_from)

            # Layer-wise directional drift
            cos = F.cosine_similarity(H_o, H_d, dim=1)
            drift = (1.0 - cos).clamp(min=0.0)

            bad = drift > THRESH
            first_bad = int(torch.argmax(bad.to(torch.int64)).item()) if bool(bad.any()) else -1
            max_bad = int(torch.argmax(drift).item())

            # Top-3 output overlap
            set_o = set(int(x) for x in top_o[:3].tolist())
            set_d = set(int(x) for x in top_d[:3].tolist())
            overlap = len(set_o & set_d) / 3.0

            records.append({
                "pi": pi,
                "label": label,
                "step": st,
                "H_o": H_o,
                "H_d": H_d,
                "top_o": top_o,
                "top_d": top_d,
                "prob_o": prob_o,
                "prob_d": prob_d,
                "drift": drift,
                "first_bad": first_bad,
                "max_bad": max_bad,
                "overlap": overlap,
            })

    # Summary metrics
    overlaps = [r["overlap"] for r in records]
    firsts = [r["first_bad"] for r in records if r["first_bad"] >= 0]
    median_first = float(np.median(firsts)) if firsts else None

    ratios = []
    jm_step0 = None

    for st in range(STEPS + 1):
        step_recs = [r for r in records if r["step"] == st]

        vecs_o = []
        vecs_d = []
        names = []

        for r in step_recs:
            vo = F.normalize(r["H_o"].reshape(-1), dim=0)
            vd = F.normalize(r["H_d"].reshape(-1), dim=0)
            vecs_o.append(vo)
            vecs_d.append(vd)
            names.append(r["label"])

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                orig_dist = 1.0 - float(torch.dot(vecs_o[i], vecs_o[j]).item())
                dna_dist = 1.0 - float(torch.dot(vecs_d[i], vecs_d[j]).item())

                if orig_dist > 1e-4:
                    ratio = dna_dist / orig_dist
                else:
                    ratio = 1.0

                ratios.append(ratio)

                if st == 0 and names[i] == "Japan" and names[j] == "Mexico":
                    jm_step0 = ratio

    min_ratio = float(np.min(ratios)) if ratios else 1.0
    avg_overlap = float(np.mean(overlaps)) * 100.0 if overlaps else 0.0

    summary = {
        "compress_from": compress_from,
        "avg_overlap": avg_overlap,
        "median_first": median_first,
        "min_ratio": min_ratio,
        "jm_step0": jm_step0,
    }

    return summary, records


# ------------------------------------------------------------
# 7. Sweep configs
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("🧪 SELECTIVE DEPTH SWEEP: keep first N layers full")
print("=" * 78)

print(
    f"{'keep_full':<10}{'AvgTop3%':<10}{'MedFirst':<10}"
    f"{'MinRatio':<10}{'JP-MX S0':<10}{'Verdict'}"
)
print("-" * 78)

summaries = []
focus_records = None

for cf in CONFIGS:
    res, recs = evaluate_config(cf)
    summaries.append((cf, res))

    if cf == FOCUS_CF:
        focus_records = recs

    med_s = "-" if res["median_first"] is None else f"L{int(res['median_first'])}"
    jm_s = "-" if res["jm_step0"] is None else f"{res['jm_step0']:.3f}"

    if cf == NL:
        verdict = "🟢 CONTROL"
    elif res["min_ratio"] >= 0.70 and res["avg_overlap"] >= 70:
        verdict = "✅ SAFE"
    elif res["min_ratio"] >= 0.50 and res["avg_overlap"] >= 50:
        verdict = "🟡 PARTIAL"
    else:
        verdict = "🔴 COLLAPSE"

    print(
        f"{cf:<10}{res['avg_overlap']:<10.1f}{med_s:<10}"
        f"{res['min_ratio']:<10.3f}{jm_s:<10}{verdict}"
    )

# ------------------------------------------------------------
# 8. Compare full DNA vs selective DNA
# ------------------------------------------------------------
full_res = None
sel_res = None
ctrl_res = None

for cf, res in summaries:
    if cf == 0:
        full_res = res
    if cf == 8:
        sel_res = res
    if cf == NL:
        ctrl_res = res

print("\n" + "=" * 78)
print("🏆 MISSION 1.16 VERDICT")
print("=" * 78)

if full_res and sel_res:
    print(f"Full DNA       : overlap={full_res['avg_overlap']:.1f}%, min_ratio={full_res['min_ratio']:.3f}")
    print(f"Selective cf=8 : overlap={sel_res['avg_overlap']:.1f}%, min_ratio={sel_res['min_ratio']:.3f}")

    if ctrl_res:
        print(f"Control no-DNA : overlap={ctrl_res['avg_overlap']:.1f}%, min_ratio={ctrl_res['min_ratio']:.3f}")

    print()

    if (
        sel_res["min_ratio"] > full_res["min_ratio"] + 0.20 and
        sel_res["avg_overlap"] > full_res["avg_overlap"] + 15.0
    ):
        print("✅ EARLY-LAYER COMPRESSION WAS A MAJOR CULPRIT.")
        print("   Layers 0-7 full rakhne se identity collapse kaafi improve hua.")

        if sel_res["min_ratio"] >= 0.70 and sel_res["avg_overlap"] >= 70:
            print("   🟢 Strong fix. Next: depth-wise DNA + physical DNA artifact.")
        else:
            print("   🟡 Partial fix. Next: contrastive identity subspace preserve karo compressed layers mein.")
    else:
        print("❌ Keeping early layers full is not enough.")
        print("   Identity collapse abhi bhi mid/late layers mein hai.")
        print("   Next: Mission 1.17 Contrastive Identity-Preserving DNA.")
else:
    print("⚠️ Required configs not found. Use CONFIGS containing 0 and 8.")

# ------------------------------------------------------------
# 9. Detailed focus report for compress_from = 8
# ------------------------------------------------------------
if focus_records is not None:
    print("\n" + "=" * 78)
    print(f"🔍 FOCUS REPORT: compress_from={FOCUS_CF}")
    print("=" * 78)

    print(
        f"{'Prompt':<9}{'Step':<5}{'FirstDrift':<11}{'MaxDrift':<9}"
        f"{'L0':<6}{'L5':<6}{'L11':<6}{'L23':<6}{'Top3':<6}{'Orig -> DNA'}"
    )
    print("-" * 95)

    for r in focus_records:
        drift = r["drift"]
        first_s = "-" if r["first_bad"] < 0 else f"L{r['first_bad']}"
        max_s = f"L{r['max_bad']}"

        o1 = decode_top(r["top_o"], 1)[0]
        d1 = decode_top(r["top_d"], 1)[0]

        print(
            f"{r['label']:<9}{r['step']:<5}{first_s:<11}{max_s:<9}"
            f"{drift[0]:<6.3f}{drift[min(5, NL-1)]:<6.3f}"
            f"{drift[min(11, NL-1)]:<6.3f}{drift[NL-1]:<6.3f}"
            f"{r['overlap']*100:<5.0f}% {o1} -> {d1}"
        )

    # Sample cement signature: Layer 11 top active dims, step 0
    print("\n" + "=" * 78)
    print(f"🧱 SAMPLE CEMENT SIGNATURE — Layer 11 Top-8 Active Dims, Step 0, cf={FOCUS_CF}")
    print("=" * 78)

    l_show = min(11, NL - 1)
    for r in focus_records:
        if r["step"] != 0:
            continue

        dims_o = torch.topk(r["H_o"][l_show].abs(), 8).indices.tolist()
        dims_d = torch.topk(r["H_d"][l_show].abs(), 8).indices.tolist()

        print(f"{r['label']:<9} ORIG dims: {dims_o}")
        print(f"{r['label']:<9} DNA  dims: {dims_d}")
        print()

    # Identity collapse detail for focus config
    print("=" * 78)
    print(f"🧩 IDENTITY COLLAPSE DETAIL — cf={FOCUS_CF}")
    print("=" * 78)

    print(f"{'Step':<5}{'Pair':<18}{'OrigDist':<10}{'DNADist':<10}{'Ratio':<8}{'Verdict'}")
    print("-" * 70)

    for st in range(STEPS + 1):
        step_recs = [r for r in focus_records if r["step"] == st]

        vecs_o = []
        vecs_d = []
        names = []

        for r in step_recs:
            vo = F.normalize(r["H_o"].reshape(-1), dim=0)
            vd = F.normalize(r["H_d"].reshape(-1), dim=0)
            vecs_o.append(vo)
            vecs_d.append(vd)
            names.append(r["label"])

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                orig_dist = 1.0 - float(torch.dot(vecs_o[i], vecs_o[j]).item())
                dna_dist = 1.0 - float(torch.dot(vecs_d[i], vecs_d[j]).item())

                if orig_dist > 1e-4:
                    ratio = dna_dist / orig_dist
                else:
                    ratio = 1.0

                pair = f"{names[i]}-{names[j]}"

                if orig_dist < 0.02:
                    verdict = "⚪ TOO SIMILAR"
                elif ratio < 0.50:
                    verdict = "🔴 COLLAPSED"
                elif ratio < 0.80:
                    verdict = "🟡 PARTIAL"
                else:
                    verdict = "🟢 OK"

                print(
                    f"{st:<5}{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}"
                    f"{ratio:<8.3f}{verdict}"
                )

# ------------------------------------------------------------
# 10. Save summary artifact
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)
summary_file = "results/mission_1_16_selective_depth_summary.npz"

cfg_arr = np.array([cf for cf, _ in summaries], dtype=np.int64)
ov_arr = np.array([res["avg_overlap"] for _, res in summaries], dtype=np.float32)
mr_arr = np.array([res["min_ratio"] for _, res in summaries], dtype=np.float32)
jm_arr = np.array(
    [res["jm_step0"] if res["jm_step0"] is not None else -1.0 for _, res in summaries],
    dtype=np.float32
)
mf_arr = np.array(
    [res["median_first"] if res["median_first"] is not None else -1.0 for _, res in summaries],
    dtype=np.float32
)

np.savez_compressed(
    summary_file,
    configs=cfg_arr,
    avg_overlap=ov_arr,
    min_ratio=mr_arr,
    jm_step0=jm_arr,
    median_first=mf_arr,
)

print("\n" + "=" * 78)
print(f"💾 Summary saved: {summary_file}")
print("=" * 78)

# ------------------------------------------------------------
# 11. Cleanup
# ------------------------------------------------------------
for h in dna_hooks:
    h.remove()

print(f"⏱️ Total time: {time.time()-t0:.1f}s")

del model
gc.collect()