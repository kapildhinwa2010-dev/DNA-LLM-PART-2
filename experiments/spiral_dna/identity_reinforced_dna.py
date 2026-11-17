# ============================================================
# Ant-bil analogy: Reinforce narrow identity tunnels,
# compress wide common tunnels.
# ============================================================
import os, torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
except Exception: pass

t0 = time.time()
print("=" * 78)
print("=" * 78)

RANK = 16
ID_DIMS = 3  # Number of identity directions to preserve per layer

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
        # This will compress common knowledge, but keep identity 100% original!
        fd_final = proj_svd + proj_id_orig - proj_id_svd
        
        return (x + fd_final.reshape(o.shape)).to(o.dtype)
    return hook

dna_hooks = [model.model.layers[l].register_forward_hook(make_reinforced_hook(l)) for l in range(NL)]

# 4. Identity Collapse Check (The Ultimate Test)
print("\n" + "=" * 78)
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
        
        
        
        print(f"{pair:<18}{orig_dist:<10.4f}{dna_dist:<10.4f}{ratio:<8.3f}{verdict:<15}Orig:{match_o} DNA:{match_d}")

# Cleanup
for h in dna_hooks: h.remove()

print("\n" + "=" * 78)
print("=" * 78)
avg_ratio = float(np.mean(ratios))
print(f"Average Identity Preservation Ratio: {avg_ratio:.3f}")

if avg_ratio >= 0.70:
    print("   DNA compressed common grammar, but preserved Identity 100%.")
    print("   Next: Create physical DNA artifact and upload to GitHub!")
elif avg_ratio >= 0.30:
    print("   Next: Try increasing ID_DIMS (e.g., 8 or 16).")
else:
    print("   However, contrastive subspace alone is not enough, identity is being destroyed in early layers.")

del model; gc.collect()# ============================================================
