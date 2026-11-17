# ============================================================
# MISSION 1.21: SPIRAL ZOOM DNA (Fractal SVD)
# Replace flat Rank-16 with 3-Level Residual Zoom
# ============================================================
import os, torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try: torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
except: pass

t0 = time.time()
print("=" * 78)
print("MISSION 1.21: SPIRAL ZOOM DNA")
print("=" * 78)

# Zoom Levels: 3 levels, 4 dimensions each = Total 12 effective dimensions
ZOOM_LEVELS = 3
RANK_PER_LEVEL = 4

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

cal_prompts = ["The capital of France is", "The capital of Japan is", 
               "The capital of Mexico is", "The chemical symbol for silver is"]
cal_inp = tok(cal_prompts, return_tensors="pt", padding=True)
cal_mask = cal_inp["attention_mask"].bool()

print("Capturing deltas for Spiral Zoom SVD...")
boxes = {l: {} for l in range(NL)}

def make_hook(l_idx):
    def hook(m, i, o):
        boxes[l_idx]["in"] = i[0].detach().float()
        boxes[l_idx]["out"] = o.detach().float()
        return o
    return hook

hooks = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in range(NL)]
with torch.no_grad(): model(**cal_inp)
for h in hooks: h.remove()

print("Computing Spiral Zoom Bases (Residual SVD)...")
spiral_bases = {}

for l in range(NL):
    delta = boxes[l]["out"] - boxes[l]["in"]
    flat = delta.reshape(-1, D)[cal_mask.reshape(-1)].numpy()
    mu = flat.mean(0)
    residual = flat - mu
    
    levels = []
    for z in range(ZOOM_LEVELS):
        # SVD on current residual
        if residual.shape[0] == 0: break
        _, _, Vt = np.linalg.svd(residual, full_matrices=False)
        Vz = torch.tensor(Vt[:RANK_PER_LEVEL], dtype=torch.float32)
        
        # Project and subtract to get next level residual (The Zoom)
        proj = ((torch.tensor(residual, dtype=torch.float32) @ Vz.T) @ Vz).numpy()
        residual = residual - proj
        
        levels.append(Vz)
        
    spiral_bases[l] = (torch.tensor(mu, dtype=torch.float32), levels)

print(f"Spiral Bases ready ({time.time()-t0:.1f}s)")

# The Spiral Zoom Hook
STATE = {"dna": False}

def make_zoom_hook(l_idx):
    def hook(m, i, o):
        if not STATE["dna"]: return o
        
        x = i[0].float()
        delta = o.float() - x
        fd = delta.reshape(-1, D)
        
        mu, levels = spiral_bases[l_idx]
        
        # Reconstruct via Spiral Zoom
        fd_rec = torch.zeros_like(fd) + mu
        
        current_residual = fd - mu
        for Vz in levels:
            proj = (current_residual @ Vz.T) @ Vz
            fd_rec = fd_rec + proj
            current_residual = current_residual - proj
            
        return (x + fd_rec.reshape(o.shape)).to(o.dtype)
    return hook

zoom_hooks = [model.model.layers[l].register_forward_hook(make_zoom_hook(l)) for l in range(NL)]

# Identity Collapse Check
print("\n" + "=" * 78)
print("SPIRAL ZOOM IDENTITY CHECK")
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

ids_list = [tok(p, return_tensors="pt").input_ids for p in eval_prompts]
vecs_o = [F.normalize(get_flow(ids, False).reshape(-1), dim=0) for ids in ids_list]
vecs_d = [F.normalize(get_flow(ids, True).reshape(-1), dim=0) for ids in ids_list]

print(f"{'Pair':<18}{'OrigDist':<10}{'DNADist':<10}{'Ratio':<8}{'Verdict'}")
print("-" * 60)

ratios = []
for i in range(len(eval_labels)):
    for j in range(i + 1, len(eval_labels)):
        orig_dist = 1.0 - float(torch.dot(vecs_o[i], vecs_o[j]).item())
        dna_dist = 1.0 - float(torch.dot(vecs_d[i], vecs_d[j]).item())
        ratio = dna_dist / orig_dist if orig_dist > 1e-4 else 1.0
        ratios.append(ratio)
        
        pair = f"{eval_labels[i]}-{eval_labels[j]}"
        verdict = "PRESERVED" if ratio >= 0.70 else ("PARTIAL" if ratio >= 0.30 else "COLLAPSED")
        print(f"{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}{ratio:<8.3f}{verdict}")

for h in zoom_hooks: h.remove()

print("\n" + "=" * 78)
print("MISSION 1.21 VERDICT")
print("=" * 78)
avg_ratio = float(np.mean(ratios))
print(f"Average Identity Preservation Ratio: {avg_ratio:.3f}")

if avg_ratio >= 0.70:
    print("BREAKTHROUGH! Spiral Zoom has fixed Identity Collapse!")
elif avg_ratio >= 0.30:
    print("Improvement! Zoom levels are partially preserving identity.")
else:
    print("Spiral Zoom is behaving like flat SVD.")

print(f"\nTime taken: {time.time()-t0:.1f}s")
del model; gc.collect()
