# ============================================================
# 🧱 MISSION 1.19: IDENTITY-REINFORCED DNA (The Reinforced Tunnels)
# Ant-bil analogy: Narrow identity tunnels ko reinforce karo,
# wide common tunnels ko compress karo.
# ============================================================
import os, torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
except Exception: pass

t0 = time.time()
print("=" * 78)
print("🧱 MISSION 1.19: IDENTITY-REINFORCED DNA")
print("=" * 78)

RANK = 16
ID_DIMS = 3  # Kitne identity directions preserve karne hain per layer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# 1. Calibration Prompts (For General SVD)
cal_prompts = ["The capital of France is", "The president of USA in 2009 was", 
               "The chemical symbol for gold is", "In Python, to sort a list you use"]
cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

# 2. Identity Prompts (For Contrastive Tunnels)
id_prompts = ["The capital of France is", "The capital of Japan is", 
              "The capital of Mexico is", "The chemical symbol for silver is"]
id_inp = tok(id_prompts, return_tensors="pt", padding=True)
id_mask = id_inp["attention_mask"].bool()

print("📐 Capturing deltas for General SVD and Identity Subspaces...")
boxes_cal = {l: {} for l in range(NL)}
boxes_id = {l: {} for l in range(NL)}

def make_hook(l_idx, is_id):
    def hook(m, i, o):
        target = boxes_id if is_id else boxes_cal
        target[l_idx]["in"] = i[0].detach().float()
        target[l_idx]["out"] = o.detach().float()
        return o
    return hook

# Run Calibration
hooks_cal = [model.model.layers[l].register_forward_hook(make_hook(l, False)) for l in range(NL)]
with torch.no_grad(): model(**cal_inp)
for h in hooks_cal: h.remove()

# Run Identity
hooks_id = [model.model.layers[l].register_forward_hook(make_hook(l, True)) for l in range(NL)]
with torch.no_grad(): model(**id_inp)
for h in hooks_id: h.remove()

print("⏳ Computing General SVD Bases + Identity Contrastive Subspaces...")
bases = {}
id_subspaces = {}

for l in range(NL):
    # --- General SVD (Common Tunnels) ---
    delta_cal = boxes_cal[l]["out"] - boxes_cal[l]["in"]
    flat_cal = delta_cal.reshape(-1, D)[cal_mask.reshape(-1)].numpy()
    mu_cal = flat_cal.mean(0)
    _, _, Vt_cal = np.linalg.svd(flat_cal - mu_cal, full_matrices=False)
    Vk = torch.tensor(Vt_cal[:RANK], dtype=torch.float32)
    mu = torch.tensor(mu_cal, dtype=torch.float32)
    bases[l] = (Vk, mu)
    
    # --- Identity Subspace (Narrow Tunnels) ---
    delta_id = boxes_id[l]["out"] - boxes_id[l]["in"] # (4, L, D)
    # Extract last valid token hidden state for each of the 4 prompts
    h_id = []
    for b in range(len(id_prompts)):
        m = id_mask[b]
        if m.any():
            h_id.append(delta_id[b][m][-1].numpy()) # Last token delta
    
    if len(h_id) >= 2:
        h_id = np.stack(h_id, axis=0) # (4, D)
        h_id_centered = h_id - h_id.mean(0)
        # SVD to find directions that separate France/Japan/Mexico/Silver
        _, _, Vt_id = np.linalg.svd(h_id_centered, full_matrices=False)
        id_subspaces[l] = torch.tensor(Vt_id[:ID_DIMS], dtype=torch.float32)
    else:
        id_subspaces[l] = torch.zeros((ID_DIMS, D), dtype=torch.float32)

print(f"✅ General Bases + Identity Subspaces ready ({time.time()-t0:.1f}s)")

# 3. The Reinforced DNA Hook
STATE = {"dna": False}

def make_reinforced_hook(l_idx):
    def hook(m, i, o):
        if not STATE["dna"]: return o
        
        x = i[0].float()
        delta = o.float() - x
        fd = delta.reshape(-1, D)
        
        Vk, mu = bases[l_idx]
        V_id = id_subspaces[l_idx]
        
        # Step A: Compress everything via General SVD (Common Tunnels)
        proj_svd = ((fd - mu) @ Vk.T) @ Vk + mu
        
        # Step B: Extract EXACT Identity Component from Original Delta
        proj_id_orig = (fd @ V_id.T) @ V_id
        
        # Step C: Extract Identity Component from SVD Compressed Delta (to avoid double counting)
        proj_id_svd = (proj_svd @ V_id.T) @ V_id
        
        # Step D: Final Delta = SVD Compressed + Original Identity - SVD's Identity
        # Isse common knowledge compress hoga, par identity 100% original rahegi!
        fd_final = proj_svd + proj_id_orig - proj_id_svd
        
        return (x + fd_final.reshape(o.shape)).to(o.dtype)
    return hook

dna_hooks = [model.model.layers[l].register_forward_hook(make_reinforced_hook(l)) for l in range(NL)]

# 4. Identity Collapse Check (The Ultimate Test)
print("\n" + "=" * 78)
print("🧩 IDENTITY COLLAPSE CHECK: Original vs Reinforced DNA")
print("=" * 78)

eval_prompts = ["The capital of Japan is", "The capital of Mexico is", "The chemical symbol for silver is"]
eval_labels = ["Japan", "Mexico", "Silver"]

def get_flow(ids, dna=False):
    STATE["dna"] = dna
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    H = torch.stack([out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)], dim=0)
    logits = out.logits[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    tk = torch.topk(probs, 3)
    STATE["dna"] = False
    return H, tk.indices

print(f"{'Pair':<18}{'OrigDist':<10}{'DNADist':<10}{'Ratio':<8}{'Verdict':<15}{'Top1 Match?'}")
print("-" * 80)

ratios = []
top1_matches = 0
total_tests = 0

ids_list = [tok(p, return_tensors="pt").input_ids for p in eval_prompts]

vecs_o = []
vecs_d = []
top1_o = []
top1_d = []

for ids in ids_list:
    H_o, tk_o = get_flow(ids, dna=False)
    H_d, tk_d = get_flow(ids, dna=True)
    
    vecs_o.append(F.normalize(H_o.reshape(-1), dim=0))
    vecs_d.append(F.normalize(H_d.reshape(-1), dim=0))
    top1_o.append(tk_o[0].item())
    top1_d.append(tk_d[0].item())

for i in range(len(eval_labels)):
    for j in range(i + 1, len(eval_labels)):
        orig_dist = 1.0 - float(torch.dot(vecs_o[i], vecs_o[j]).item())
        dna_dist = 1.0 - float(torch.dot(vecs_d[i], vecs_d[j]).item())
        
        ratio = dna_dist / orig_dist if orig_dist > 1e-4 else 1.0
        ratios.append(ratio)
        
        pair = f"{eval_labels[i]}-{eval_labels[j]}"
        
        if ratio < 0.30: verdict = "🔴 COLLAPSED"
        elif ratio < 0.70: verdict = "🟡 PARTIAL"
        else: verdict = "🟢 PRESERVED"
        
        match_o = "✅" if top1_o[i] == top1_o[j] else "❌" # Should be ❌ (different answers)
        match_d = "✅" if top1_d[i] == top1_d[j] else "❌"
        
        print(f"{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}{ratio:<8.3f}{verdict:<15}Orig:{match_o} DNA:{match_d}")

# Cleanup
for h in dna_hooks: h.remove()

print("\n" + "=" * 78)
print("🏆 MISSION 1.19 VERDICT")
print("=" * 78)
avg_ratio = float(np.mean(ratios))
print(f"Average Identity Preservation Ratio: {avg_ratio:.3f}")

if avg_ratio >= 0.70:
    print("🎉 MASSIVE BREAKTHROUGH! Tera 'Reinforced Tunnels' idea kaam kar gaya!")
    print("   DNA ne common grammar ko compress kiya, par Identity ko 100% bacha liya.")
    print("   Next: Physical DNA Artifact banao aur GitHub par daalo!")
elif avg_ratio >= 0.30:
    print("⚠️ PARTIAL SUCCESS. Identity collapse kaafi kam hua hai.")
    print("   Next: ID_DIMS badha kar dekho (e.g., 8 ya 16).")
else:
    print("❌ Identity abhi bhi collapse ho rahi hai.")
    print("   Matlab contrastive subspace akela kaafi nahi hai, early layers mein hi identity destroy ho rahi hai.")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")
del model; gc.collect()# ============================================================
# 🧬 MISSION 1.19: CONTRASTIVE IDENTITY PROBE MAP
# Random probes ki jagah concept-specific contrastive directions use karo
# France vs Japan vs Mexico ke identity paths ko separate karo
# ============================================================

import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    cpu = os.cpu_count() or 1
    torch.set_num_threads(max(1, cpu - 1))
except Exception:
    pass

t0 = time.time()
print("=" * 80)
print("🧬 MISSION 1.19: CONTRASTIVE IDENTITY PROBE MAP")
print("=" * 80)

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Mexico is",
]

LABELS = ["France", "Japan", "Mexico"]

# Probe amplitudes
EPS1 = 1e-2
EPS2 = 3e-2

# How many contrastive directions to test per concept pair
N_CONTRASTIVE = 4

print(f"⚙️ Model: {MODEL_NAME}")
print(f"⚙️ Prompts: {PROMPTS}")
print(f"⚙️ Contrastive directions per pair: {N_CONTRASTIVE}")

# ------------------------------------------------------------
# 1. LOAD MODEL
# ------------------------------------------------------------
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

print(f"✅ Model loaded | layers={NL} | hidden_dim={D}")

# ------------------------------------------------------------
# 2. BASELINE RUNS + EXTRACT HIDDEN STATES
# ------------------------------------------------------------
def get_last_hidden_and_logits(ids):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    H = torch.stack(
        [out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)],
        dim=0,
    )  # (NL, D)
    logits = out.logits[0, -1, :].float()
    return H, logits

baselines = []
print("\n🧱 Running baselines...")
for pi, prompt in enumerate(PROMPTS):
    ids = tok(prompt, return_tensors="pt").input_ids
    H_base, logits_base = get_last_hidden_and_logits(ids)
    top1_base = int(torch.argmax(logits_base).item())
    
    baselines.append({
        "prompt": prompt,
        "label": LABELS[pi],
        "ids": ids,
        "H": H_base,
        "logits": logits_base,
        "top1": top1_base,
    })
    print(f"   [{pi}] '{prompt}' | top1 id={top1_base}")

print(f"✅ Baselines ready ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 3. CREATE CONTRASTIVE IDENTITY DIRECTIONS
# ------------------------------------------------------------
print("\n🧬 Creating contrastive identity directions...")

contrastive_probes = {}  # {layer_idx: [probe_vectors]}

for l in range(NL):
    layer_probes = []
    
    # For each pair of concepts, create contrastive direction
    for i in range(len(baselines)):
        for j in range(i + 1, len(baselines)):
            H_i = baselines[i]["H"][l]  # (D,)
            H_j = baselines[j]["H"][l]  # (D,)
            
            # Contrastive direction: what makes i different from j?
            diff = H_i - H_j
            diff_norm = diff / diff.norm()
            
            # Add N_CONTRASTIVE variations with small random perturbations
            for k in range(N_CONTRASTIVE):
                if k == 0:
                    # Pure contrastive direction
                    probe = diff_norm
                else:
                    # Add small random noise to explore nearby directions
                    noise = torch.randn(D, dtype=torch.float32) * 0.1
                    probe = diff_norm + noise
                    probe = probe / probe.norm()
                
                layer_probes.append(probe)
    
    contrastive_probes[l] = layer_probes

total_probes = sum(len(contrastive_probes[l]) for l in range(NL))
print(f"✅ Created {total_probes} contrastive probes across {NL} layers")
print(f"   Per layer: {len(contrastive_probes[0])} probes")
print(f"   Approx forward passes: {len(PROMPTS) * total_probes * 2}")

# ------------------------------------------------------------
# 4. PROBE INJECTION HOOKS
# ------------------------------------------------------------
PROBE_STATE = {
    "active_layer": None,
    "u": None,
    "eps": EPS1,
}

def make_pre_hook(l_idx):
    def hook(module, args):
        if PROBE_STATE["active_layer"] != l_idx:
            return None
        x = args[0].clone()
        x[:, -1, :] = x[:, -1, :] + PROBE_STATE["eps"] * PROBE_STATE["u"]
        return (x,) + args[1:]
    return hook

hooks = [
    model.model.layers[l].register_forward_pre_hook(make_pre_hook(l))
    for l in range(NL)
]

# ------------------------------------------------------------
# 5. RUN CONTRASTIVE PROBE SWEEP
# ------------------------------------------------------------
records = []

print("\n📡 Running contrastive probe sweep...")
print(f"{'Concept':<10}{'Layer':<7}{'Probe':<7}{'FinalGain':<12}{'MaxResp':<10}{'Noise':<10}{'Flip'}")
print("-" * 80)

for pi, base in enumerate(baselines):
    ids = base["ids"]
    Hb = base["H"]
    top1_base = base["top1"]
    label = base["label"]
    
    for l in range(NL):
        for pidx, u in enumerate(contrastive_probes[l]):
            # ---------------- EPS1 ----------------
            PROBE_STATE["active_layer"] = l
            PROBE_STATE["u"] = u
            PROBE_STATE["eps"] = EPS1
            
            with torch.no_grad():
                out1 = model(ids, output_hidden_states=True)
            
            H1 = torch.stack(
                [out1.hidden_states[m + 1][0, -1, :].float() for m in range(NL)],
                dim=0,
            )
            
            delta1 = H1 - Hb
            gain1 = delta1.norm(dim=1) / EPS1
            
            logits1 = out1.logits[0, -1, :].float()
            top1_probe1 = int(torch.argmax(logits1).item())
            flip1 = int(top1_probe1 != top1_base)
            
            # ---------------- EPS2 ----------------
            PROBE_STATE["eps"] = EPS2
            
            with torch.no_grad():
                out2 = model(ids, output_hidden_states=True)
            
            H2 = torch.stack(
                [out2.hidden_states[m + 1][0, -1, :].float() for m in range(NL)],
                dim=0,
            )
            
            delta2 = H2 - Hb
            gain2 = delta2.norm(dim=1) / EPS2
            
            # Non-linear residual
            expected_delta2 = (EPS2 / EPS1) * delta1
            nonlinear_residual = (delta2 - expected_delta2).norm(dim=1) / EPS2
            noise_scalar = float(nonlinear_residual.mean().item())
            
            max_resp_layer = int(torch.argmax(gain1).item())
            final_gain = float(gain1[-1].item())
            
            rec = {
                "prompt_idx": pi,
                "label": label,
                "inject_layer": l,
                "probe_idx": pidx,
                "gain1": gain1.detach().cpu(),
                "gain2": gain2.detach().cpu(),
                "noise": noise_scalar,
                "max_resp_layer": max_resp_layer,
                "final_gain": final_gain,
                "flip": flip1,
            }
            
            records.append(rec)
            
            # Print first probe of each layer
            if pidx == 0:
                print(
                    f"{label:<10}{l:<7}{pidx:<7}"
                    f"{final_gain:<12.5f}{max_resp_layer:<10}"
                    f"{noise_scalar:<10.5f}{'🔴' if flip1 else '🟢'}"
                )

# Turn off probing
PROBE_STATE["active_layer"] = None
PROBE_STATE["u"] = None

for h in hooks:
    h.remove()

print(f"\n✅ Contrastive probe sweep done ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 6. LAYER-LEVEL SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 CONTRASTIVE LAYER SONAR SUMMARY")
print("=" * 80)
print(
    f"{'InjectLayer':<13}{'AvgFinalGain':<15}{'AvgMaxResp':<13}"
    f"{'AvgNoise':<12}{'FlipRate%'}"
)
print("-" * 80)

layer_stats = []

for l in range(NL):
    rs = [r for r in records if r["inject_layer"] == l]
    if not rs:
        continue
    
    avg_final_gain = float(np.mean([r["final_gain"] for r in rs]))
    avg_max_resp = float(np.mean([r["max_resp_layer"] for r in rs]))
    avg_noise = float(np.mean([r["noise"] for r in rs]))
    flip_rate = float(np.mean([r["flip"] for r in rs]) * 100.0)
    
    layer_stats.append((l, avg_final_gain, avg_max_resp, avg_noise, flip_rate))
    
    print(
        f"{l:<13}{avg_final_gain:<15.5f}{avg_max_resp:<13.2f}"
        f"{avg_noise:<12.5f}{flip_rate:.1f}"
    )

# ------------------------------------------------------------
# 7. IDENTITY FINGERPRINT CHECK (CONTRASTIVE)
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("🧩 CONTRASTIVE IDENTITY FINGERPRINT CHECK")
print("=" * 80)

features = []
feature_labels = []

for pi, base in enumerate(baselines):
    rs = [r for r in records if r["prompt_idx"] == pi]
    
    # Flatten all gain1 vectors
    feat = torch.cat([r["gain1"] for r in rs], dim=0)
    feat = F.normalize(feat.unsqueeze(0), dim=1).squeeze(0)
    
    features.append(feat)
    feature_labels.append(base["label"])

print(f"{'Pair':<38}{'CosineDist':<14}{'Verdict'}")
print("-" * 80)

pair_distances = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        cos = float(torch.dot(features[i], features[j]).item())
        dist = 1.0 - cos
        pair_distances.append(dist)
        
        if dist < 0.005:
            verdict = "⚪ VERY SIMILAR"
        elif dist < 0.02:
            verdict = "🟡 CLOSE"
        elif dist < 0.05:
            verdict = "🟠 MODERATE"
        else:
            verdict = "🟢 SEPARABLE"
        
        pair = f"{feature_labels[i]} vs {feature_labels[j]}"
        print(f"{pair:<38}{dist:<14.6f}{verdict}")

# ------------------------------------------------------------
# 8. TOP IDENTITY-CRITICAL LAYERS
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("🎯 TOP IDENTITY-CRITICAL LAYERS (Highest Flip Rate)")
print("=" * 80)

by_flip = sorted(layer_stats, key=lambda x: x[4], reverse=True)

for l, g, mr, n, fr in by_flip[:10]:
    if fr > 0:
        print(f"   Inject L{l:02d} -> flip rate {fr:.1f}%, final gain {g:.5f}, noise {n:.5f}")

if all(fr == 0 for _, _, _, _, fr in layer_stats):
    print("   ⚠️ No flips detected. Model very stable under contrastive probes.")
    print("   Try larger EPS or more aggressive contrastive directions.")

# ------------------------------------------------------------
# 9. SAVE ARTIFACT
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)
sonar_file = "results/mission_1_19_contrastive_sonar.npz"

inject_layer_arr = np.array([r["inject_layer"] for r in records], dtype=np.int16)
probe_idx_arr = np.array([r["probe_idx"] for r in records], dtype=np.int16)
prompt_idx_arr = np.array([r["prompt_idx"] for r in records], dtype=np.int16)

gain1_arr = np.stack([r["gain1"].numpy() for r in records], axis=0).astype(np.float32)
gain2_arr = np.stack([r["gain2"].numpy() for r in records], axis=0).astype(np.float32)
noise_arr = np.array([r["noise"] for r in records], dtype=np.float32)
max_resp_arr = np.array([r["max_resp_layer"] for r in records], dtype=np.int16)
final_gain_arr = np.array([r["final_gain"] for r in records], dtype=np.float32)
flip_arr = np.array([r["flip"] for r in records], dtype=np.int8)

prompt_labels_arr = np.array([base["prompt"] for base in baselines], dtype=object)

np.savez_compressed(
    sonar_file,
    inject_layer=inject_layer_arr,
    probe_idx=probe_idx_arr,
    prompt_idx=prompt_idx_arr,
    gain1=gain1_arr,
    gain2=gain2_arr,
    noise=noise_arr,
    max_resp_layer=max_resp_arr,
    final_gain=final_gain_arr,
    flip=flip_arr,
    prompt_labels=prompt_labels_arr,
)

size_kb = os.path.getsize(sonar_file) / 1024
print("\n" + "=" * 80)
print(f"💾 Contrastive sonar map saved: {sonar_file}")
print(f"📦 Size: {size_kb:.2f} KB")
print("=" * 80)

# ------------------------------------------------------------
# 10. FINAL VERDICT
# ------------------------------------------------------------
avg_noise = float(np.mean([r["noise"] for r in records]))
avg_flip = float(np.mean([r["flip"] for r in records]) * 100.0)
avg_pair_dist = float(np.mean(pair_distances)) if pair_distances else 0.0

print("\n🏆 MISSION 1.19 VERDICT")
print("-" * 80)
print(f"Average non-linear noise         : {avg_noise:.6f}")
print(f"Average top-1 flip rate          : {avg_flip:.2f}%")
print(f"Average identity fingerprint dist: {avg_pair_dist:.6f}")

print("\nInterpretation:")
if avg_pair_dist > 0.05:
    print("🟢 SUCCESS! Contrastive probes ne identity paths ko SEPARABLE bana diya!")
    print("   France/Japan/Mexico ke internal routing patterns ab alag dikh rahe hain.")
    print("   Next: Routing-Aware DNA banao jo in identity-critical paths ko preserve kare.")
elif avg_pair_dist > 0.02:
    print("🟠 MODERATE improvement. Identity partially separable.")
    print("   More contrastive probes ya larger EPS try karo.")
else:
    print("🔴 Identity abhi bhi CLOSE hai.")
    print("   Contrastive directions kaafi nahi. Try:")
    print("   - Full sequence probes (last token nahi)")
    print("   - Layer-specific identity subspace extraction")
    print("   - Larger EPS amplitudes")

if avg_flip > 20:
    print("\n⚠️ High flip rate! Probes too strong. Reduce EPS.")
elif avg_flip == 0:
    print("\n✅ Model stable. No flips under contrastive probes.")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")

del model
gc.collect() ==============================================================================
🧱 MISSION 1.19: IDENTITY-REINFORCED DNA
==============================================================================
Loading weights: 100% 290/290 [00:04<00:00, 90.58it/s]📐 Capturing deltas for General SVD and Identity Subspaces...
⏳ Computing General SVD Bases + Identity Contrastive Subspaces...
✅ General Bases + Identity Subspaces ready (13.7s)

==============================================================================
🧩 IDENTITY COLLAPSE CHECK: Original vs Reinforced DNA
==============================================================================
Pair              OrigDist  DNADist   Ratio   Verdict        Top1 Match?
--------------------------------------------------------------------------------
Japan-Mexico      0.1119    0.0004    0.003   🔴 COLLAPSED    Orig:✅ DNA:✅
Japan-Silver      0.3111    0.2869    0.922   🟢 PRESERVED    Orig:❌ DNA:❌
Mexico-Silver     0.3467    0.2826    0.815   🟢 PRESERVED    Orig:❌ DNA:❌

==============================================================================
🏆 MISSION 1.19 VERDICT
==============================================================================
Average Identity Preservation Ratio: 0.580
⚠️ PARTIAL SUCCESS. Identity collapse kaafi kam hua hai.
   Next: ID_DIMS badha kar dekho (e.g., 8 ya 16).

⏱️ Total time: 18.5s
64 ================================================================================
🧬 MISSION 1.19: CONTRASTIVE IDENTITY PROBE MAP
================================================================================
⚙️ Model: Qwen/Qwen2.5-0.5B-Instruct
⚙️ Prompts: ['The capital of France is', 'The capital of Japan is', 'The capital of Mexico is']
⚙️ Contrastive directions per pair: 4
Loading weights: 100% 290/290 [00:01<00:00, 242.82it/s]✅ Model loaded | layers=24 | hidden_dim=896

🧱 Running baselines...
   [0] 'The capital of France is' | top1 id=12095
   [1] 'The capital of Japan is' | top1 id=7407
   [2] 'The capital of Mexico is' | top1 id=7407
✅ Baselines ready (3.6s)

🧬 Creating contrastive identity directions...
✅ Created 288 contrastive probes across 24 layers
   Per layer: 12 probes
   Approx forward passes: 1728

📡 Running contrastive probe sweep...
Concept   Layer  Probe  FinalGain   MaxResp   Noise     Flip
--------------------------------------------------------------------------------
France    0      0      50.44689    23        0.66077   🟢
France    1      0      10.51660    23        0.01277   🟢
France    2      0      9.68214     23        0.00636   🟢
France    3      0      9.46907     23        0.00633   🟢
France    4      0      8.29138     23        0.00389   🟢
France    5      0      9.32642     23        0.00547   🟢
France    6      0      10.69858    23        0.00318   🟢
France    7      0      9.08875     23        0.00342   🟢
France    8      0      6.51173     23        0.00323   🟢
France    9      0      7.10552     23        0.00237   🟢
France    10     0      8.74894     23        0.00225   🟢
France    11     0      9.72487     23        0.00333   🟢
France    12     0      10.43862    23        0.00295   🟢
France    13     0      12.05559    23        0.00316   🟢
France    14     0      10.81436    23        0.00261   🟢
France    15     0      12.39639    23        0.00229   🟢
France    16     0      11.28329    23        0.00171   🟢
France    17     0      14.07753    23        0.00232   🟢
France    18     0      12.50199    23        0.00132   🟢
France    19     0      9.17066     23        0.00095   🟢
France    20     0      8.81715     23        0.00079   🟢
France    21     0      6.50230     23        0.00064   🟢
France    22     0      5.24260     23        0.00040   🟢
France    23     0      4.99201     23        0.00033   🟢
Japan     0      0      37.98639    23        0.60779   🟢
Japan     1      0      10.54198    23        0.01480   🟢
Japan     2      0      8.77155     23        0.00748   🟢
Japan     3      0      10.40137    23        0.00875   🟢
Japan     4      0      9.57039     23        0.00378   🟢
Japan     5      0      13.34649    23        0.00997   🟢
Japan     6      0      7.31480     23        0.00335   🟢
Japan     7      0      7.43739     23        0.00458   🟢
Japan     8      0      13.40084    23        0.00369   🟢
Japan     9      0      5.66392     23        0.00266   🟢
Japan     10     0      8.46850     23        0.00248   🟢
Japan     11     0      11.98734    23        0.00327   🟢
Japan     12     0      12.95379    23        0.00246   🟢
Japan     13     0      14.29067    23        0.00323   🟢
Japan     14     0      13.20554    23        0.00262   🟢
Japan     15     0      15.65458    23        0.00263   🟢
Japan     16     0      13.19218    23        0.00188   🟢
Japan     17     0      13.18766    23        0.00300   🟢
Japan     18     0      12.23827    23        0.00160   🟢
Japan     19     0      11.01029    23        0.00105   🟢
Japan     20     0      9.32280     23        0.00092   🟢
Japan     21     0      6.86769     23        0.00060   🟢
Japan     22     0      5.39177     23        0.00046   🟢
Japan     23     0      5.27865     23        0.00037   🟢
Mexico    0      0      63.24669    23        0.75515   🟢
Mexico    1      0      12.35206    23        0.01228   🟢
Mexico    2      0      9.16118     23        0.00732   🟢
Mexico    3      0      7.94769     23        0.00675   🟢
Mexico    4      0      12.35012    23        0.00431   🟢
Mexico    5      0      10.83887    23        0.00609   🟢
Mexico    6      0      13.28879    23        0.00368   🟢
Mexico    7      0      10.60778    23        0.00354   🟢
Mexico    8      0      10.30840    23        0.00278   🟢
Mexico    9      0      8.18512     23        0.00211   🟢
Mexico    10     0      10.01656    23        0.00244   🟢
Mexico    11     0      11.47154    23        0.00365   🟢
Mexico    12     0      12.25681    23        0.00245   🟢
Mexico    13     0      13.65622    23        0.00272   🟢
Mexico    14     0      12.36634    23        0.00251   🟢
Mexico    15     0      15.03575    23        0.00256   🟢
Mexico    16     0      13.07484    23        0.00199   🟢
Mexico    17     0      13.40780    23        0.00197   🟢
Mexico    18     0      14.21867    23        0.00156   🟢
Mexico    19     0      10.28864    23        0.00101   🟢
Mexico    20     0      9.46704     23        0.00089   🟢
Mexico    21     0      6.32242     23        0.00058   🟢
Mexico    22     0      5.66032     23        0.00049   🟢
Mexico    23     0      5.50029     23        0.00045   🟢

✅ Contrastive probe sweep done (720.9s)

================================================================================
📊 CONTRASTIVE LAYER SONAR SUMMARY
================================================================================
InjectLayer  AvgFinalGain   AvgMaxResp   AvgNoise    FlipRate%
--------------------------------------------------------------------------------
0            50.87583       23.00        0.67918     0.0
1            8.88312        23.00        0.00933     0.0
2            11.61651       23.00        0.00715     0.0
3            7.76774        23.00        0.00527     0.0
4            8.58103        23.00        0.00389     0.0
5            9.54912        23.00        0.00441     0.0
6            7.89232        23.00        0.00299     0.0
7            9.67858        23.00        0.00355     0.0
8            8.62720        23.00        0.00314     0.0
9            9.15868        23.00        0.00299     0.0
10           10.30122       23.00        0.00307     0.0
11           10.25730       23.00        0.00311     0.0
12           9.63661        23.00        0.00251     0.0
13           10.29069       23.00        0.00255     0.0
14           10.46099       23.00        0.00223     0.0
15           11.64669       23.00        0.00206     0.0
16           9.45880        23.00        0.00149     0.0
17           10.14954       23.00        0.00158     0.0
18           8.94930        23.00        0.00117     0.0
19           7.64837        23.00        0.00095     0.0
20           6.92263        23.00        0.00082     0.0
21           5.58666        23.00        0.00064     0.0
22           4.98344        23.00        0.00049     0.0
23           4.44354        23.00        0.00033     0.0

================================================================================
🧩 CONTRASTIVE IDENTITY FINGERPRINT CHECK
================================================================================
Pair                                  CosineDist    Verdict
--------------------------------------------------------------------------------
France vs Japan                       0.016458      🟡 CLOSE
France vs Mexico                      0.017806      🟡 CLOSE
Japan vs Mexico                       0.043881      🟠 MODERATE

================================================================================
🎯 TOP IDENTITY-CRITICAL LAYERS (Highest Flip Rate)
================================================================================
   ⚠️ No flips detected. Model very stable under contrastive probes.
   Try larger EPS or more aggressive contrastive directions.

================================================================================
💾 Contrastive sonar map saved: results/mission_1_19_contrastive_sonar.npz
📦 Size: 86.99 KB
================================================================================

🏆 MISSION 1.19 VERDICT
--------------------------------------------------------------------------------
Average non-linear noise         : 0.031038
Average top-1 flip rate          : 0.00%
Average identity fingerprint dist: 0.026048

Interpretation:
🟠 MODERATE improvement. Identity partially separable.
   More contrastive probes ya larger EPS try karo.

✅ Model stable. No flips under contrastive probes.

⏱️ Total time: 720.9s
0# ============================================================
# 🧱 MISSION 1.20: HIGH-RESOLUTION IDENTITY-REINFORCED DNA
# SVD ko zyada prompts dekar Japan/Mexico jaise close concepts ko separate karna
# ============================================================
import os, torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try: torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
except: pass

t0 = time.time()
print("=" * 78)
print("🧱 MISSION 1.20: HIGH-RES IDENTITY-REINFORCED DNA")
print("=" * 78)

RANK = 16
ID_DIMS = 6  # Thoda badha diya kyunki ab zyada concepts hain

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# 1. General Calibration (Common Tunnels)
cal_prompts = ["The capital of France is", "The president of USA in 2009 was", 
               "The chemical symbol for gold is", "In Python, to sort a list you use"]
cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

# 2. HIGH-RES Identity Prompts (Narrow Tunnels)
# Humne 12 diverse concepts daale hain taaki SVD Japan aur Mexico ko alag kar sake
id_prompts = [
    "The capital of France is", "The capital of Japan is", "The capital of Mexico is",
    "The capital of UK is", "The capital of Germany is", "The capital of India is",
    "The chemical symbol for silver is", "The chemical symbol for iron is",
    "The largest planet is Jupiter", "The smallest planet is Mercury",
    "Water boils at 100 degrees", "The square root of 144 is 12"
]
id_inp = tok(id_prompts, return_tensors="pt", padding=True)
id_mask = id_inp["attention_mask"].bool()

print("📐 Capturing deltas for General SVD and High-Res Identity Subspaces...")
boxes_cal = {l: {} for l in range(NL)}
boxes_id = {l: {} for l in range(NL)}

def make_hook(l_idx, is_id):
    def hook(m, i, o):
        target = boxes_id if is_id else boxes_cal
        target[l_idx]["in"] = i[0].detach().float()
        target[l_idx]["out"] = o.detach().float()
        return o
    return hook

hooks_cal = [model.model.layers[l].register_forward_hook(make_hook(l, False)) for l in range(NL)]
with torch.no_grad(): model(**cal_inp)
for h in hooks_cal: h.remove()

hooks_id = [model.model.layers[l].register_forward_hook(make_hook(l, True)) for l in range(NL)]
with torch.no_grad(): model(**id_inp)
for h in hooks_id: h.remove()

print("⏳ Computing High-Res Contrastive Subspaces...")
bases = {}
id_subspaces = {}

for l in range(NL):
    # General SVD
    delta_cal = boxes_cal[l]["out"] - boxes_cal[l]["in"]
    flat_cal = delta_cal.reshape(-1, D)[cal_mask.reshape(-1)].numpy()
    mu_cal = flat_cal.mean(0)
    _, _, Vt_cal = np.linalg.svd(flat_cal - mu_cal, full_matrices=False)
    bases[l] = (torch.tensor(Vt_cal[:RANK], dtype=torch.float32), torch.tensor(mu_cal, dtype=torch.float32))
    
    # High-Res Identity Subspace
    delta_id = boxes_id[l]["out"] - boxes_id[l]["in"] 
    h_id = []
    for b in range(len(id_prompts)):
        m = id_mask[b]
        if m.any():
            h_id.append(delta_id[b][m][-1].numpy()) 
            
    if len(h_id) >= 2:
        h_id = np.stack(h_id, axis=0) 
        h_id_centered = h_id - h_id.mean(0)
        _, _, Vt_id = np.linalg.svd(h_id_centered, full_matrices=False)
        id_subspaces[l] = torch.tensor(Vt_id[:ID_DIMS], dtype=torch.float32)
    else:
        id_subspaces[l] = torch.zeros((ID_DIMS, D), dtype=torch.float32)

print(f"✅ High-Res Subspaces ready ({time.time()-t0:.1f}s)")

# 3. The Reinforced DNA Hook
STATE = {"dna": False}
def make_reinforced_hook(l_idx):
    def hook(m, i, o):
        if not STATE["dna"]: return o
        x = i[0].float()
        delta = o.float() - x
        fd = delta.reshape(-1, D)
        Vk, mu = bases[l_idx]
        V_id = id_subspaces[l_idx]
        
        proj_svd = ((fd - mu) @ Vk.T) @ Vk + mu
        proj_id_orig = (fd @ V_id.T) @ V_id
        proj_id_svd = (proj_svd @ V_id.T) @ V_id
        
        fd_final = proj_svd + proj_id_orig - proj_id_svd
        return (x + fd_final.reshape(o.shape)).to(o.dtype)
    return hook

dna_hooks = [model.model.layers[l].register_forward_hook(make_reinforced_hook(l)) for l in range(NL)]

# 4. The Ultimate Test
print("\n" + "=" * 78)
print("🧩 HIGH-RES IDENTITY COLLAPSE CHECK")
print("=" * 78)

eval_prompts = ["The capital of Japan is", "The capital of Mexico is", "The capital of France is"]
eval_labels = ["Japan", "Mexico", "France"]

def get_flow(ids, dna=False):
    STATE["dna"] = dna
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    H = torch.stack([out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)], dim=0)
    STATE["dna"] = False
    return H

print(f"{'Pair':<18}{'OrigDist':<10}{'DNADist':<10}{'Ratio':<8}{'Verdict'}")
print("-" * 60)

ids_list = [tok(p, return_tensors="pt").input_ids for p in eval_prompts]
vecs_o = [F.normalize(get_flow(ids, False).reshape(-1), dim=0) for ids in ids_list]
vecs_d = [F.normalize(get_flow(ids, True).reshape(-1), dim=0) for ids in ids_list]

ratios = []
for i in range(len(eval_labels)):
    for j in range(i + 1, len(eval_labels)):
        orig_dist = 1.0 - float(torch.dot(vecs_o[i], vecs_o[j]).item())
        dna_dist = 1.0 - float(torch.dot(vecs_d[i], vecs_d[j]).item())
        ratio = dna_dist / orig_dist if orig_dist > 1e-4 else 1.0
        ratios.append(ratio)
        
        pair = f"{eval_labels[i]}-{eval_labels[j]}"
        verdict = "🟢 PRESERVED" if ratio >= 0.70 else ("🟡 PARTIAL" if ratio >= 0.30 else "🔴 COLLAPSED")
        print(f"{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}{ratio:<8.3f}{verdict}")

for h in dna_hooks: h.remove()

print("\n" + "=" * 78)
print("🏆 MISSION 1.20 VERDICT")
print("=" * 78)
avg_ratio = float(np.mean(ratios))
print(f"Average Identity Preservation Ratio: {avg_ratio:.3f}")

if avg_ratio >= 0.70:
    print("🎉 PERFECT! High-Res SVD ne Japan aur Mexico ko bhi alag kar diya!")
    print("   Tera 'Reinforced Tunnels' idea ab 100% complete aur proven hai.")
    print("   Next: Isko ek Physical DNA Artifact (.dna file) mein pack karo!")
elif avg_ratio >= 0.30:
    print("⚠️ Improvement hua hai, par abhi bhi thoda collapse hai.")
else:
    print("❌ Identity abhi bhi mix ho rahi hai.")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")
del model; gc.collect() ==============================================================================
🧱 MISSION 1.20: HIGH-RES IDENTITY-REINFORCED DNA
==============================================================================
Loading weights: 100% 290/290 [00:02<00:00, 167.25it/s]📐 Capturing deltas for General SVD and High-Res Identity Subspaces...
⏳ Computing High-Res Contrastive Subspaces...
✅ High-Res Subspaces ready (11.6s)

==============================================================================
🧩 HIGH-RES IDENTITY COLLAPSE CHECK
==============================================================================
Pair              OrigDist  DNADist   Ratio   Verdict
------------------------------------------------------------
Japan-Mexico      0.1119    0.0004    0.004   🔴 COLLAPSED
Japan-France      0.1176    0.0028    0.023   🔴 COLLAPSED
Mexico-France     0.1018    0.0048    0.047   🔴 COLLAPSED

==============================================================================
🏆 MISSION 1.20 VERDICT
==============================================================================
Average Identity Preservation Ratio: 0.025
❌ Identity abhi bhi mix ho rahi hai.

⏱️ Total time: 17.2s
64