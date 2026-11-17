# ============================================================
#  MISSION 1.18: FULL LAYER-WISE BROADBAND SONAR MAP
# Ant-bil analogy:
# - Har layer par known probe/frequency daalo
# - Har downstream layer par response naapo
# - Width = gain
# - Length = response delay/max layer
# - Breadth = active dimensions
# - Noise = non-linear residual across scales
# - Full map save karo for routing-aware DNA
# ============================================================

import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# CPU threads
try:
    cpu = os.cpu_count() or 1
    torch.set_num_threads(max(1, cpu - 1))
except Exception:
    pass

t0 = time.time()
print("=" * 80)
print(" MISSION 1.18: FULL LAYER-WISE BROADBAND SONAR MAP")
print("=" * 80)

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Prompts = different colored cement / different concept paths
PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Mexico is",
]

# Number of random coded probes per layer.
# 4 = balanced. 2 = faster. 8 = better map but slower.
PROBES_PER_LAYER = 4

# Two amplitudes for non-linearity/noise estimate.
EPS1 = 1e-2
EPS2 = 3e-2

# For top active dimensions per response layer.
TOPK_DIMS = 8

# Relative threshold for counting active response dimensions.
ACTIVE_REL_THRESH = 0.05

# Probe mode:
# "coded"  = random broadband probes, practical
# "basis_subset" = exact one-hot probes on subset dims, more interpretable but slower
PROBE_MODE = "coded"
BASIS_SUBSET_DIMS = 16  # used only if PROBE_MODE == "basis_subset"

SAVE_FULL_DELTA = True  # saves full delta tensors as float16 for later map reconstruction

print(f"⚙️ Model: {MODEL_NAME}")
print(f"⚙️ Prompts: {len(PROMPTS)}")
print(f"⚙️ Probes/layer: {PROBES_PER_LAYER}")
print(f"⚙️ Probe mode: {PROBE_MODE}")
print(f"⚙️ EPS scales: {EPS1}, {EPS2}")
print(f"⚙️ Save full delta: {SAVE_FULL_DELTA}")

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
# 2. CREATE PROBES
# ------------------------------------------------------------
torch.manual_seed(1234)

if PROBE_MODE == "coded":
    # Random normalized directions = broadband coded sonar
    probes = []
    for l in range(NL):
        layer_probes = []
        for _ in range(PROBES_PER_LAYER):
            u = torch.randn(D, dtype=torch.float32)
            u = u / u.norm()
            layer_probes.append(u)
        probes.append(layer_probes)
    print(" Created random coded probes.")

elif PROBE_MODE == "basis_subset":
    # Exact one-hot probes on selected dimensions
    dims = list(range(min(BASIS_SUBSET_DIMS, D)))
    probes = []
    for l in range(NL):
        layer_probes = []
        for dim in dims:
            u = torch.zeros(D, dtype=torch.float32)
            u[dim] = 1.0
            layer_probes.append(u)
        probes.append(layer_probes)
    print(f" Created basis_subset probes on dims: {dims}")

else:
    raise ValueError("PROBE_MODE must be 'coded' or 'basis_subset'")

actual_probes_per_layer = len(probes[0])
total_forwards = len(PROMPTS) * NL * actual_probes_per_layer * 2
print(f" Approx forward passes: {total_forwards}")

# ------------------------------------------------------------
# 3. BASELINE RUNS
# ------------------------------------------------------------
def get_last_hidden_and_logits(ids):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states[0] = embedding output
    # hidden_states[l+1] = output of transformer layer l
    H = torch.stack(
        [out.hidden_states[l + 1][0, -1, :].float() for l in range(NL)],
        dim=0,
    )  # (NL, D)

    logits = out.logits[0, -1, :].float()  # (vocab,)
    return H, logits

baselines = []

print("\n Running baselines...")
for pi, prompt in enumerate(PROMPTS):
    ids = tok(prompt, return_tensors="pt").input_ids
    H_base, logits_base = get_last_hidden_and_logits(ids)
    top1_base = int(torch.argmax(logits_base).item())

    baselines.append({
        "prompt": prompt,
        "ids": ids,
        "H": H_base,
        "logits": logits_base,
        "top1": top1_base,
    })

    print(f"   [{pi}] '{prompt}' | baseline top1 id={top1_base}")

print(f"✅ Baselines ready ({time.time()-t0:.1f}s)")

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

        # args[0] is hidden_states entering this layer
        x = args[0].clone()

        # Inject probe only at last token position for speed/locality
        x[:, -1, :] = x[:, -1, :] + PROBE_STATE["eps"] * PROBE_STATE["u"]

        return (x,) + args[1:]
    return hook

hooks = [
    model.model.layers[l].register_forward_pre_hook(make_pre_hook(l))
    for l in range(NL)
]

# ------------------------------------------------------------
# 5. RUN FULL SONAR SWEEP
# ------------------------------------------------------------
records = []

print("\n Running full layer-wise sonar sweep...")
print(f"{'Prompt':<10}{'Layer':<7}{'Probe':<7}{'FinalGain':<12}{'MaxResp':<10}{'Noise':<10}{'Flip'}")
print("-" * 80)

for pi, base in enumerate(baselines):
    ids = base["ids"]
    Hb = base["H"]
    top1_base = base["top1"]

    # short label
    words = base["prompt"].split()
    label = words[-2] if len(words) >= 2 else f"P{pi}"

    for l in range(NL):
        for pidx, u in enumerate(probes[l]):

            # ---------------- EPS1 ----------------
            PROBE_STATE["active_layer"] = l
            PROBE_STATE["u"] = u
            PROBE_STATE["eps"] = EPS1

            with torch.no_grad():
                out1 = model(ids, output_hidden_states=True)

            H1 = torch.stack(
                [out1.hidden_states[m + 1][0, -1, :].float() for m in range(NL)],
                dim=0,
            )  # (NL, D)

            delta1 = H1 - Hb  # (NL, D)
            gain1 = delta1.norm(dim=1) / EPS1  # (NL,)

            # Top active dimensions per response layer
            delta_abs1 = delta1.abs()
            top_val1, top_idx1 = torch.topk(delta_abs1, TOPK_DIMS, dim=1)

            # Active breadth: how many response layers/dims matter
            max_gain1 = float(gain1.max().item())
            if max_gain1 > 0:
                active_layer_count = int((gain1 > ACTIVE_REL_THRESH * max_gain1).sum().item())
            else:
                active_layer_count = 0

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

            # Non-linear residual:
            # If locally linear: delta2 ≈ (EPS2/EPS1) * delta1
            expected_delta2 = (EPS2 / EPS1) * delta1
            nonlinear_residual = (delta2 - expected_delta2).norm(dim=1) / EPS2
            noise_scalar = float(nonlinear_residual.mean().item())

            # Max response layer = length/delay proxy
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
                "nonlinear_residual": nonlinear_residual.detach().cpu(),
                "max_resp_layer": max_resp_layer,
                "final_gain": final_gain,
                "active_layer_count": active_layer_count,
                "flip": flip1,
                "top_idx1": top_idx1.detach().cpu().to(torch.int32),
                "top_val1": top_val1.detach().cpu(),
            }

            if SAVE_FULL_DELTA:
                rec["delta1"] = delta1.detach().cpu()
                rec["delta2"] = delta2.detach().cpu()

            records.append(rec)

            if pidx == 0:
                print(
                    f"{label:<10}{l:<7}{pidx:<7}"
                    f"{final_gain:<12.5f}{max_resp_layer:<10}"
                    f"{noise_scalar:<10.5f}{'' if flip1 else ''}"
                )

        # Optional progress per layer
        # print(f"   finished inject layer {l}")

# Turn off probing
PROBE_STATE["active_layer"] = None
PROBE_STATE["u"] = None

# Remove hooks
for h in hooks:
    h.remove()

print(f"\n✅ Sonar sweep done ({time.time()-t0:.1f}s)")

# ------------------------------------------------------------
# 6. LAYER-LEVEL SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 80)
print(" LAYER SONAR SUMMARY")
print("=" * 80)
print(
    f"{'InjectLayer':<13}{'AvgFinalGain':<15}{'AvgMaxResp':<13}"
    f"{'AvgNoise':<12}{'AvgBreadth':<13}{'FlipRate%'}"
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
    avg_breadth = float(np.mean([r["active_layer_count"] for r in rs]))
    flip_rate = float(np.mean([r["flip"] for r in rs]) * 100.0)

    layer_stats.append((l, avg_final_gain, avg_max_resp, avg_noise, avg_breadth, flip_rate))

    print(
        f"{l:<13}{avg_final_gain:<15.5f}{avg_max_resp:<13.2f}"
        f"{avg_noise:<12.5f}{avg_breadth:<13.2f}{flip_rate:.1f}"
    )

# ------------------------------------------------------------
# 7. TOP PATHS: HIGHWAYS, NOISY TUNNELS, LONG-RANGE PATHS
# ------------------------------------------------------------
if layer_stats:
    by_gain = sorted(layer_stats, key=lambda x: x[1], reverse=True)
    by_noise = sorted(layer_stats, key=lambda x: x[3], reverse=True)
    by_delay = sorted(layer_stats, key=lambda x: abs(x[2] - x[0]), reverse=True)
    by_breadth = sorted(layer_stats, key=lambda x: x[4], reverse=True)

    print("\n️ TOP WIDE HIGHWAYS (highest final gain):")
    for l, g, mr, n, b, fr in by_gain[:5]:
        print(f"   Inject L{l:02d} -> final gain {g:.5f}, max response ~L{mr:.1f}, noise {n:.5f}")

    print("\n TOP NOISY / NON-LINEAR TUNNELS:")
    for l, g, mr, n, b, fr in by_noise[:5]:
        print(f"   Inject L{l:02d} -> noise {n:.5f}, final gain {g:.5f}")

    print("\n TOP LONG-RANGE PATHS (largest delay):")
    for l, g, mr, n, b, fr in by_delay[:5]:
        print(f"   Inject L{l:02d} -> max response L{mr:.1f}, delay {abs(mr-l):.1f} layers")

    print("\n TOP BROAD/FAN-OUT PATHS:")
    for l, g, mr, n, b, fr in by_breadth[:5]:
        print(f"   Inject L{l:02d} -> active response layers {b:.2f}, final gain {g:.5f}")

# ------------------------------------------------------------
# 8. IDENTITY FINGERPRINT CHECK
#    Do France/Japan/Mexico produce different full sonar fingerprints?
# ------------------------------------------------------------
print("\n" + "=" * 80)
print(" IDENTITY SONAR FINGERPRINT CHECK")
print("=" * 80)

features = []
feature_labels = []

for pi, base in enumerate(baselines):
    rs = [r for r in records if r["prompt_idx"] == pi]

    # Flatten all gain1 vectors: inject_layer × probe × response_layer
    feat = torch.cat([r["gain1"] for r in rs], dim=0)
    feat = F.normalize(feat.unsqueeze(0), dim=1).squeeze(0)

    features.append(feat)
    feature_labels.append(base["prompt"])

print(f"{'Pair':<48}{'CosineDist':<14}{'Verdict'}")
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
            verdict = " CLOSE"
        elif dist < 0.05:
            verdict = " MODERATE"
        else:
            verdict = " SEPARABLE"

        pair = f"{feature_labels[i][:22]} vs {feature_labels[j][:22]}"
        print(f"{pair:<48}{dist:<14.6f}{verdict}")

# ------------------------------------------------------------
# 9. SAVE FULL SONAR ARTIFACT
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)
sonar_file = "results/mission_1_18_full_sonar_map.npz"

inject_layer_arr = np.array([r["inject_layer"] for r in records], dtype=np.int16)
probe_idx_arr = np.array([r["probe_idx"] for r in records], dtype=np.int16)
prompt_idx_arr = np.array([r["prompt_idx"] for r in records], dtype=np.int16)

gain1_arr = np.stack([r["gain1"].numpy() for r in records], axis=0).astype(np.float32)
gain2_arr = np.stack([r["gain2"].numpy() for r in records], axis=0).astype(np.float32)
noise_arr = np.array([r["noise"] for r in records], dtype=np.float32)
max_resp_arr = np.array([r["max_resp_layer"] for r in records], dtype=np.int16)
final_gain_arr = np.array([r["final_gain"] for r in records], dtype=np.float32)
active_count_arr = np.array([r["active_layer_count"] for r in records], dtype=np.int16)
flip_arr = np.array([r["flip"] for r in records], dtype=np.int8)

top_idx_arr = np.stack([r["top_idx1"].numpy() for r in records], axis=0).astype(np.int32)
top_val_arr = np.stack([r["top_val1"].numpy() for r in records], axis=0).astype(np.float32)

prompt_labels_arr = np.array([base["prompt"] for base in baselines], dtype=object)

save_dict = {
    "inject_layer": inject_layer_arr,
    "probe_idx": probe_idx_arr,
    "prompt_idx": prompt_idx_arr,
    "gain1": gain1_arr,
    "gain2": gain2_arr,
    "noise": noise_arr,
    "max_resp_layer": max_resp_arr,
    "final_gain": final_gain_arr,
    "active_layer_count": active_count_arr,
    "flip": flip_arr,
    "top_idx": top_idx_arr,
    "top_val": top_val_arr,
    "prompt_labels": prompt_labels_arr,
    "config": np.array([
        MODEL_NAME,
        str(PROBES_PER_LAYER),
        str(EPS1),
        str(EPS2),
        PROBE_MODE,
    ], dtype=object),
}

if SAVE_FULL_DELTA:
    # Save as float16 to keep file size reasonable
    delta1_arr = np.stack([r["delta1"].numpy() for r in records], axis=0).astype(np.float16)
    delta2_arr = np.stack([r["delta2"].numpy() for r in records], axis=0).astype(np.float16)
    save_dict["delta1"] = delta1_arr
    save_dict["delta2"] = delta2_arr

np.savez_compressed(sonar_file, **save_dict)

size_kb = os.path.getsize(sonar_file) / 1024
print("\n" + "=" * 80)
print(f" Full sonar map saved: {sonar_file}")
print(f" Size: {size_kb:.2f} KB")
print("=" * 80)

# ------------------------------------------------------------
# 10. FINAL VERDICT
# ------------------------------------------------------------
avg_noise = float(np.mean([r["noise"] for r in records]))
avg_flip = float(np.mean([r["flip"] for r in records]) * 100.0)
avg_pair_dist = float(np.mean(pair_distances)) if pair_distances else 0.0

print("\n MISSION 1.18 VERDICT")
print("-" * 80)
print(f"Average non-linear noise       : {avg_noise:.6f}")
print(f"Average top-1 flip rate        : {avg_flip:.2f}%")
print(f"Average identity fingerprint dist: {avg_pair_dist:.6f}")

print("\nInterpretation:")
if avg_flip < 15 and avg_noise < 0.03:
    print("✅ Model locally quite stable under these probe scales.")
    print("   Sonar map is reliable for identifying important routing paths.")
elif avg_flip < 40:
    print(" Moderate non-linearity / sensitivity.")
    print("   More probes or smaller EPS may improve map stability.")
else:
    print(" High sensitivity / probe too strong or model very non-linear at this scale.")
    print("   Reduce EPS1/EPS2 or increase PROBES_PER_LAYER with smaller amplitude.")

if avg_pair_dist > 0.02:
    print(" Concept paths are separable in original model sonar fingerprint.")
    print("   France/Japan/Mexico use distinguishable internal routing patterns.")
elif avg_pair_dist > 0.005:
    print(" Concept paths are partially separable.")
    print("   Need more probes or layer-focused identity subspace mapping.")
else:
    print(" Concept paths are very similar in this probe map.")
    print("   May need concept-aligned probes instead of random probes.")

print("\n NEXT STEP:")
print("   Use this sonar map to build Routing-Aware DNA:")
print("   - High-gain identity paths -> preserve")
print("   - Low-behavior high-variance paths -> compress")
print("   - Noisy non-linear tunnels -> handle carefully / distill")

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")

# Cleanup
del model
