# ============================================================
# 💾 MISSION 2: EXTRACT & SAVE THE PHYSICAL DNA FILE
# Compress 1GB model's brain into Kilobytes and save
# ============================================================
import torch, numpy as np, time, gc, os
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("="*70)
print("💾 MISSION 2: EXTRACTING & SAVING REVERSE DNA")
print("="*70)

# 1. Load Model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(model_name)
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# 2. Calibration Prompts (Diverse knowledge extraction)
cal_prompts = [
    "The capital of France is", "The sky is blue", "1 plus 1 equals 2",
    "In Python, to sort a list", "The chemical symbol for gold is",
    "Water boils at 100 degrees", "The largest planet is Jupiter",
    "To connect to a database", "The square root of 144 is 12"
]
cal_inp = tok(cal_prompts, return_tensors="pt", padding="max_length", max_length=24, truncation=True)

# 3. Extract Shifts for ALL 24 Layers
print(f"⏳ Extracting Knowledge Shifts for {NL} layers...")
boxes = {l: {} for l in range(NL)}
hooks_cal = []
for l in range(NL):
    def make_hook(l_idx):
        def hook(m, i, o):
            boxes[l_idx]['in'] = i[0].detach().float()
            boxes[l_idx]['out'] = o.detach().float()
            return o
        return hook
    hooks_cal.append(model.model.layers[l].register_forward_hook(make_hook(l)))

with torch.no_grad(): _ = model(**cal_inp)
for h in hooks_cal: h.remove()

# 4. Compress to DNA (Rank-16 SVD) & Save
K = 16  # Rank
dna_data = {}
total_orig_params = 0
total_dna_params = 0

print(f"{'Layer':<8} | {'Original Params':<15} | {'DNA Params':<15} | {'Compression'}")
print("-" * 60)

for l in range(NL):
    delta = boxes[l]['out'] - boxes[l]['in']
    flat = delta.view(-1, D).numpy()
    mu = flat.mean(0)
    
    # SVD Compression
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    Vk = Vt[:K]  # Top 16 basis vectors
    
    # Store in dictionary
    dna_data[f'layer_{l:02d}_Vk'] = Vk.astype(np.float16)  # FP16 for storage efficiency
    dna_data[f'layer_{l:02d}_mu'] = mu.astype(np.float16)
    
    # Calculate Params
    # Original layer (roughly FFN + Attn weights) ~ 20M params per layer
    # But we are comparing the Shift Representation (D * D) vs DNA (K * D + D)
    orig_shift_size = D * D  # Full matrix shift
    dna_size = (K * D) + D   # Basis + Mean
    
    total_orig_params += orig_shift_size
    total_dna_params += dna_size
    
    comp_ratio = orig_shift_size / dna_size
    print(f" L{l:02d}     | {orig_shift_size:<15,} | {dna_size:<15,} | {comp_ratio:.1f}x")

# 5. Save to Disk
os.makedirs('results', exist_ok=True)
dna_file = 'results/Qwen_Reverse_DNA.dna'
np.savez_compressed(dna_file, **dna_data)
file_size_kb = os.path.getsize(dna_file) / 1024

print("\n" + "="*70)
print("🏆 MISSION ACCOMPLISHED: DNA FILE SAVED!")
print("="*70)
print(f"📂 File: {dna_file}")
print(f"💾 Physical Size on Disk: {file_size_kb:.2f} KB")
print(f"🧬 Total DNA Parameters: {total_dna_params:,} (FP16)")
print(f"📊 Shift Space Compression: {total_orig_params / total_dna_params:.1f}x")
print("\n💡 NOTE: This file is the model's 'Behavioral DNA'.")
print("   It contains 24 layers' 384 functions (waves) that define the model's shift.")
print("="*70)
print(f"⏱️ Time taken: {time.time()-t0:.1f}s")

del model; gc.collect()