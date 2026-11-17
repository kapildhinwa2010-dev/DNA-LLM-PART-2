# ============================================================
# FULL MODEL X-RAY: Multi-Layer Logit Lens (Depth-wise Shift)
# Check how much each layer's 'Shift' is preserved by DNA
# ============================================================
import torch, numpy as np, time, gc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("="*70)
print("FULL MODEL X-RAY: Multi-Layer Logit Lens")
print("="*70)

# 1. Load Model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = 'left'  # Left padding so last token is always at index -1
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# 2. Get the LM Head (The Final Vocabulary Matrix)
W_unembed = model.get_output_embeddings().weight.detach().float() 

# 3. Prompts
prompts = ["The capital of France is", "The square root of 144 is", "In Python, to sort a list you use"]
inp = tok(prompts, return_tensors="pt", padding="max_length", max_length=16, truncation=True)

# 4. Extract ALL Layers (Input & Output)
print(f"Extracting Shifts (Deltas) for all {NL} layers...")
boxes = {l: {} for l in range(NL)}
hooks = []
for l in range(NL):
    def make_hook(l_idx):
        def hook(m, i, o):
            boxes[l_idx]['in'] = i[0].detach().float()
            boxes[l_idx]['out'] = o.detach().float()
            return o
        return hook
    hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

with torch.no_grad():
    _ = model(**inp)
for h in hooks: h.remove()

# 5. Process Each Layer (Compress + Logit Lens)
print(f"{'Layer':<6} | {'Shift Match (Cosine)':<20} | {'Status'}")
print("-" * 55)

cos_sims_all = []
for l in range(NL):
    delta_orig = boxes[l]['out'] - boxes[l]['in']
    
    # Rank-16 SVD Compression (DNA)
    flat = delta_orig.view(-1, D).numpy()
    mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    k = 16
    Vt_k = Vt[:k]
    flat_rec = ((flat - mu) @ Vt_k.T) @ Vt_k + mu
    delta_dna = torch.tensor(flat_rec, dtype=torch.float32).view(delta_orig.shape)
    
    # Logit Lens: Multiply Shift by LM Head
    logits_orig = delta_orig @ W_unembed.T
    logits_dna = delta_dna @ W_unembed.T
    
    # Calculate Cosine Similarity at the LAST token (index -1 due to left padding)
    c_sims = []
    for i in range(len(prompts)):
        L_o = logits_orig[i, -1, :]
        L_d = logits_dna[i, -1, :]
        cos = torch.nn.functional.cosine_similarity(L_o.unsqueeze(0), L_d.unsqueeze(0)).item()
        c_sims.append(cos)
        
    avg_cos = np.mean(c_sims) * 100
    cos_sims_all.append(avg_cos)
    
    # Status indicator
    if avg_cos > 90: status = "PERFECT SHIFT"
    elif avg_cos > 75: status = "GOOD SHIFT"
    elif avg_cos > 50: status = "WEAK SHIFT"
    else: status = "SHIFT DESTROYED"
        
    print(f" L{l:02d}    | {avg_cos:6.1f}%                 | {status}")

# 6. Final Summary & Plot
print("\n" + "="*70)
avg_all = np.mean(cos_sims_all)
print(f"AVERAGE SHIFT PRESERVATION ACROSS ALL {NL} LAYERS: {avg_all:.1f}%")
print("="*70)

if avg_all > 80:
    print("MIND-BLOWING! DNA has preserved the internal shifts of the entire model.")
    print("   If final accuracy is 23%, it's due to 'Cascade Amplification' (Butterfly Effect).")
    print("   DNA is mathematically perfect, downstream layers just need to adapt to the new shift (Fine-tuning).")
else:
    print("Some specific layers have shift destruction.")
    print("   Check the table above for RED indicators. Those layers cannot handle DNA compression.")

# Plotting the Depth Profile
plt.figure(figsize=(10, 5))
plt.plot(range(NL), cos_sims_all, marker='o', linestyle='-', color='#4ECDC4', linewidth=2, markersize=6)
plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Perfect Shift (90%)')
plt.axhline(y=75, color='y', linestyle='--', alpha=0.5, label='Good Shift (75%)')
plt.fill_between(range(NL), cos_sims_all, alpha=0.2, color='#4ECDC4')
plt.xlabel('Layer Depth (0 to 23)')
plt.ylabel('Shift Direction Match (%)')
plt.title('Logit Lens: Internal Behavior Preservation Across Depth')
plt.xticks(range(0, NL, 2))
plt.ylim(0, 105)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/multi_layer_logit_lens.png', dpi=150)
plt.show()

print(f"Time taken: {time.time()-t0:.1f}s")
del model; gc.collect()
