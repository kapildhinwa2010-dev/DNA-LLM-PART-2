# ============================================================
# 🔬 LOGIT LENS: Layer Shift × LM Head = Asli Behavior
# Testing the User's Intuition: Matrix Multiplication Shift
# ============================================================
import torch, numpy as np, time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("="*70)
print("🔬 LOGIT LENS TEST (User's Intuition: Matrix Mult Shift)")
print("="*70)

# 1. Load Model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYER = 11
D = model.config.hidden_size

# 2. Get the LM Head (The Final Vocabulary Matrix)
# This is the matrix that converts hidden state to words (logits)
W_unembed = model.get_output_embeddings().weight.detach().float() 
print(f"✅ LM Head loaded: Shape {tuple(W_unembed.shape)} (Vocab × Hidden)")

# 3. Prompts
prompts = [
    "The capital of France is", 
    "The square root of 144 is", 
    "In Python, to sort a list you use"
]
inp = tok(prompts, return_tensors="pt", padding="max_length", max_length=16, truncation=True)

# 4. Extract Layer 11 Shift (Delta)
box = {}
def hook(m, i, o):
    box['in'] = i[0].detach().float()
    box['out'] = o.detach().float()
    return o

h = model.model.layers[LAYER].register_forward_hook(hook)
with torch.no_grad():
    _ = model(**inp)
h.remove()

h_in = box['in']
h_out = box['out']
delta_orig = h_out - h_in  # Asli Shift (B, L, D)
print(f"✅ Extracted Original Shift (Delta) for {len(prompts)} prompts.")

# 5. DNA Compression (Rank 16 SVD on the fly)
flat = delta_orig.view(-1, D).numpy()
mu = flat.mean(0)
_, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
k = 16
Vt_k = Vt[:k]
flat_rec = ((flat - mu) @ Vt_k.T) @ Vt_k + mu
delta_dna = torch.tensor(flat_rec, dtype=torch.float32).view(h_in.shape)
print(f"✅ Compressed to Rank-{k} DNA.")

# 6. TUMHARA IDEA: MATRIX MULTIPLICATION (Logit Lens)
# Delta (B, L, D) @ LM_Head.T (D, Vocab) -> Logits (B, L, Vocab)
print("⏳ Applying Logit Lens (Multiplying Shift by LM Head)...")
logits_orig = delta_orig @ W_unembed.T
logits_dna = delta_dna @ W_unembed.T

# 7. Evaluate at the LAST token (where the prediction happens)
attn_mask = inp['attention_mask']
last_idx = attn_mask.sum(dim=1) - 1 

print("\n" + "="*70)
print("📊 RESULTS: Asli Behavior Match (Direction & Top Words)")
print("="*70)

cos_sims = []
for i, prompt in enumerate(prompts):
    idx = last_idx[i].item()
    L_orig = logits_orig[i, idx, :]
    L_dna = logits_dna[i, idx, :]

    # Cosine Similarity (Direction Match)
    cos = torch.nn.functional.cosine_similarity(L_orig.unsqueeze(0), L_dna.unsqueeze(0)).item()
    cos_sims.append(cos)

    # Top 5 Words (Semantic Match)
    top5_orig_ids = torch.topk(L_orig, 5).indices
    top5_dna_ids = torch.topk(L_dna, 5).indices
    
    # Decode tokens (clean up subwords like 'Ġ' or '▁')
    words_orig = [tok.decode([t]).strip() for t in top5_orig_ids]
    words_dna = [tok.decode([t]).strip() for t in top5_dna_ids]

    overlap = len(set(words_orig) & set(words_dna))

    print(f"\n📝 Prompt: '{prompt}'")
    print(f"   📐 Direction Match (Cosine): {cos*100:.1f}%")
    print(f"   🧠 Original Top 5: {words_orig}")
    print(f"   🧬 DNA Top 5:      {words_dna}")
    print(f"   🎯 Overlap: {overlap}/5 words same!")

avg_cos = np.mean(cos_sims)
print("\n" + "="*70)
print(f"🏆 AVERAGE DIRECTION MATCH: {avg_cos*100:.1f}%")
print("="*70)

if avg_cos > 0.85:
    print("🎉 BREAKTHROUGH! TUMHARA INTUITION 100% SAHI THA!")
    print("   Layer's 'Shift' (Behavior) is 85%+ preserved.")
    print("   Hamara 23% wala metric ek 'Step Function Trap' tha.")
    print("   DNA successfully encoded the knowledge!")
elif avg_cos > 0.60:
    print("⚠️ PARTIAL MATCH: Direction roughly same, but magnitudes have shifted.")
    print("   A bit of calibration (scaling factor) can fix this.")
else:
    print("❌ SHIFT DESTROYED: DNA ne layer ka asli shift hi badal diya.")
    print("   Knowledge was not truly compressed at this rank.")

print(f"⏱️ Time taken: {time.time()-t0:.1f}s")
del model; gc.collect()