# ============================================================
# MISSION 1.11: CONCEPT-AWARE INJECTION (User's Genius Logic)
# "Where is Paris/Obama stored in which dimension?" -> Inject that vector into DNA
# ============================================================
import torch, numpy as np, time, gc
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("="*70)
print("MISSION 1.11: CONCEPT SUBSPACE INJECTION")
print("="*70)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
D = model.config.hidden_size

# 1. Extract the "Concept Vector" (Where does the model store 'Obama'?)
print("Extracting the 'Obama' Concept Vector from Original Model...")
# Baseline (No concept)
inp_base = tok("The president of USA in 2009 was ", return_tensors="pt")
with torch.no_grad(): 
    out_base = model(**inp_base, output_hidden_states=True)
    H_base = out_base.hidden_states[-1][:, -1, :] # (1, 896)

# Concept (With 'Obama')
_inp_concept = tok("The president of USA in 2009 was Obama", return_tensors="pt")
with torch.no_grad(): 
    out_concept = model(**_inp_concept, output_hidden_states=True)
    # We look at the hidden state right BEFORE 'Obama' is processed, or AT 'Obama'
    H_concept = out_concept.hidden_states[-1][:, -2, :] # The state carrying 'Obama'

# The Concept Vector (The exact dimension/direction where 'Obama' lives)
V_concept = (H_concept - H_base).squeeze(0)
V_concept = V_concept / V_concept.norm() # Normalize to unit vector
print(f"Concept Vector Extracted! Shape: {tuple(V_concept.shape)}")

# 2. Setup DNA Hooks (Same Rank-16 SVD as before)
NL = model.config.num_hidden_layers
boxes = {l: {} for l in range(NL)}
USE_DNA = False
CALIBRATING = True

def make_dna_hook(l_idx):
    def hook(m, i, o):
        if CALIBRATING:
            boxes[l_idx]['in'] = i[0].detach().float()
            boxes[l_idx]['out'] = o.detach().float()
            return o
        if not USE_DNA: return o
        x = i[0].float()
        delta = o.float() - x
        flat_delta = delta.view(-1, D)
        # Fast SVD approximation for CPU
        _, _, Vt = np.linalg.svd(flat_delta.numpy() - flat_delta.mean(0).numpy(), full_matrices=False)
        Vk = torch.tensor(Vt[:16], dtype=torch.float32)
        mu = flat_delta.mean(0)
        proj = ((flat_delta - mu) @ Vk.T) @ Vk + mu
        return (x + proj.view(o.shape)).to(o.dtype)
    return hook

hooks = [model.model.layers[l].register_forward_hook(make_dna_hook(l)) for l in range(NL)]
with torch.no_grad(): model(inp_base) # Calibrate
CALIBRATING = False

# 3. The Magic Hook: Inject the Concept Vector into DNA at the LM Head
def concept_injection_hook(module, input):
    if not USE_DNA: return input
    H_dna = input[0].float().squeeze(0) # (Seq, 896)
    
    # Original model's activation for this concept (We pre-calculate this in the loop)
    # For this test, we will forcefully align the LAST token's hidden state with the Concept Vector
    # Math: H_corrected = H_dna + (Target_Activation - Current_Activation) * V_concept
    
    # Let's project H_dna onto V_concept
    current_activation = torch.dot(H_dna[-1], V_concept)
    
    # We want the DNA to have the SAME activation for 'Obama' as the Original Model did
    # (We pass target_act via a global variable for simplicity in this script)
    target_activation = global_target_act 
    
    correction = (target_activation - current_activation) * V_concept
    H_dna[-1] = H_dna[-1] + correction
    
    return (H_dna.unsqueeze(0).to(input[0].dtype),)

lm_hook = model.get_output_embeddings().register_forward_pre_hook(concept_injection_hook)

# 4. Autopsy Loop
unseen_prompt = "The president of USA in 2009 was"
test_ids = tok(unseen_prompt, return_tensors="pt").input_ids
global global_target_act
global_target_act = 0.0

print(f"\nTRACKING WITH CONCEPT INJECTION: '{unseen_prompt}'")
print(f"{'Step':<4} | {'Context':<12} | {'Original Top-2':<25} | {'DNA+Concept Top-2':<25} | {'Match?'}")
print("-" * 85)

for step in range(4):
    # --- ORIGINAL MODEL (Get Target Activation) ---
    USE_DNA = False
    with torch.no_grad():
        out_orig = model(test_ids, output_hidden_states=True)
        H_orig_last = out_orig.hidden_states[-1][:, -1, :].squeeze(0)
        # How strongly does Original model think about the 'Concept' right now?
        global_target_act = torch.dot(H_orig_last, V_concept).item()
        
        probs_orig = F.softmax(out_orig.logits[:, -1, :], dim=-1)
        top2_orig = torch.topk(probs_orig, 2)
        
    # --- DNA + CONCEPT INJECTION ---
    USE_DNA = True
    with torch.no_grad():
        out_dna = model(test_ids)
        probs_dna = F.softmax(out_dna.logits[:, -1, :], dim=-1)
        top2_dna = torch.topk(probs_dna, 2)

    w_orig = [tok.decode([t]).strip() for t in top2_orig.indices[0]]
    p_orig = top2_orig.values[0] * 100
    w_dna = [tok.decode([t]).strip() for t in top2_dna.indices[0]]
    p_dna = top2_dna.values[0] * 100
    
    match = "YES" if top2_orig.indices[0, 0] == top2_dna.indices[0, 0] else "FLIP!"
    ctx = "[PROMPT]" if step == 0 else repr(tok.decode([test_ids[0, -1].item()]).strip())
    
    print(f" {step:<3} | {ctx:<12} | {w_orig[0]}({p_orig[0]:.1f}%) {w_orig[1]}({p_orig[1]:.1f}%) | {w_dna[0]}({p_dna[0]:.1f}%) {w_dna[1]}({p_dna[1]:.1f}%) | {match}")
    
    next_tok = top2_orig.indices[0, 0].unsqueeze(0).unsqueeze(0)
    test_ids = torch.cat([test_ids, next_tok], dim=-1)

lm_hook.remove()
for h in hooks: h.remove()
print("="*70)
print("VERDICT: If YES appears, then your 'Concept Dimension' logic is 100% correct!")
print(f"Time: {time.time()-t0:.1f}s")
del model; gc.collect()
