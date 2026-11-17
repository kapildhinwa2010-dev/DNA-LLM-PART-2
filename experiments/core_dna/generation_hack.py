# ============================================================
# MISSION 1: THE GENERATION HACK (Logit Sharpening)
# Can DNA's gibberish become fluent text without training?
# ============================================================
import torch, numpy as np, time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor

t0 = time.time()
print("="*70)
print("MISSION 1: GENERATION HACK (Sharpening the DNA Blade)")
print("="*70)

# 1. Load Model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
NL = model.config.num_hidden_layers
D = model.config.hidden_size

# 2. Calibration (Extract DNA Basis for all 24 layers)
print("Calibrating DNA Basis (Rank-16) for all 24 layers...")
cal_prompts = ["The capital of France is", "The sky is blue", "1 plus 1 equals 2"]
cal_inp = tok(cal_prompts, return_tensors="pt", padding="max_length", max_length=16, truncation=True)

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

# Compute SVD Basis
bases = {}
for l in range(NL):
    delta = boxes[l]['out'] - boxes[l]['in']
    flat = delta.view(-1, D).numpy()
    mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    # Store as torch tensors for fast hook execution
    bases[l] = (torch.tensor(Vt[:16], dtype=torch.float32), torch.tensor(mu, dtype=torch.float32))
print(f"DNA Basis Extracted for {NL} layers ({time.time()-t0:.1f}s)")

# 3. The DNA Hook (Applies Rank-16 compression on the fly during generation)
def make_dna_hook(l):
    Vk, mu = bases[l]
    def hook(module, input, output):
        x = input[0].float()
        delta = output.float() - x
        flat_delta = delta.view(-1, D)
        # Project to Rank-16
        proj = ((flat_delta - mu) @ Vk.T) @ Vk + mu
        new_out = x + proj.view(output.shape)
        return new_out.to(output.dtype)
    return hook

# 4. The Sharpening Hack (Logits Processor)
class SharpenLogits(LogitsProcessor):
    def __init__(self, alpha): self.alpha = alpha
    def __call__(self, input_ids, scores): 
        return scores * self.alpha  # Multiply logits to sharpen the peak!

# 5. Test Prompts
test_prompts = ["The capital of France is", "To sort a list in Python, use"]
test_inp = tok(test_prompts, return_tensors="pt", padding=True, return_attention_mask=True)

def generate_text(logits_processor=None):
    gen_kwargs = {
        "max_new_tokens": 10, 
        "do_sample": False, # Greedy decoding (argmax)
        "pad_token_id": tok.pad_token_id
    }
    if logits_processor:
        gen_kwargs["logits_processor"] = logits_processor
        
    with torch.no_grad():
        out = model.generate(**test_inp, **gen_kwargs)
    return tok.batch_decode(out, skip_special_tokens=True)

# --- TEST 1: BASELINE (Original Model) ---
print("\n" + "="*70)
print("TEST 1: BASELINE (Original Model, No DNA)")
res_orig = generate_text()
for p, r in zip(test_prompts, res_orig): print(f"   {p} -> {r[len(p):].strip()}")

# --- TEST 2: DNA RAW (Hooks Active, No Sharpening) ---
print("\nTEST 2: DNA RAW (Compressed, No Sharpening -> The 23% Trap)")
hooks_dna = [model.model.layers[l].register_forward_hook(make_dna_hook(l)) for l in range(NL)]
res_dna = generate_text()
for p, r in zip(test_prompts, res_dna): print(f"   {p} -> {r[len(p):].strip()}")

# --- TEST 3: DNA + SHARPENING (The Hack: Alpha = 2.0) ---
print("\nTEST 3: DNA + SHARPENING (Alpha = 2.0)")
res_sharp2 = generate_text(LogitsProcessorList([SharpenLogits(2.0)]))
for p, r in zip(test_prompts, res_sharp2): print(f"   {p} -> {r[len(p):].strip()}")

# --- TEST 4: DNA + EXTREME SHARPENING (The Hack: Alpha = 5.0) ---
print("\nTEST 4: DNA + EXTREME SHARPENING (Alpha = 5.0)")
res_sharp5 = generate_text(LogitsProcessorList([SharpenLogits(5.0)]))
for p, r in zip(test_prompts, res_sharp5): print(f"   {p} -> {r[len(p):].strip()}")

# Remove hooks
for h in hooks_dna: h.remove()

print("\n" + "="*70)
print("HACK VERDICT:")
print("   If Test 3 or 4 shows correct words (Paris, sorted) instead of gibberish,")
print("   it means DNA's knowledge was alive, but probabilities were flat!")
print("="*70)
print(f"Time taken: {time.time()-t0:.1f}s")
del model; gc.collect()
