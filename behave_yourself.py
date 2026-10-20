# ============================================================
# 🧠 OPTION B: BEHAVIOR COMPRESSIBILITY PROOF (Activations vs Weights)
# Kya "Knowledge in Action" (Activations) compressible hai?
# ============================================================
import numpy as np, os, gc, time, torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("🧠 THE MOMENT OF TRUTH: Are Activations Compressible?")
print("="*80)

# ---- PHASE 0: Load Model & Tokenizer ----
print("\n🦙 Loading TinyLlama 1.1B (FP16)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# ---- PHASE 1: Generate Activations (The "Knowledge in Action") ----
print("\n📝 PHASE 1: Generating Diverse Prompts to extract Activations...")
# 8 diverse prompts to activate different "knowledge" areas
prompts = [
    "The capital of France is",
    "In Python, to sort a list you use",
    "The theory of relativity was proposed by",
    "Photosynthesis converts sunlight into",
    "The square root of 144 is",
    "In a chess game, the knight moves in an",
    "The largest planet in our solar system is",
    "To connect to a database in SQL, you write"
]

# Tokenize (pad to 256 tokens so we get exactly 8 * 256 = 2048 rows)
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=256, truncation=True)

print("⏳ Running forward pass to capture Hidden States (Layer 10)...")
with torch.no_grad():
    # output_hidden_states=True gives us the activations at every layer
    outputs = model(**inputs, output_hidden_states=True)

# Extract Layer 10 activations (middle layer, usually richest in semantics)
# Shape: (batch=8, seq_len=256, hidden_dim=2048)
activations = outputs.hidden_states[10].float().numpy() 

# Reshape to 2D matrix: (2048 rows of context, 2048 columns of features)
A_matrix = activations.reshape(-1, 2048)
print(f"✅ Activations Matrix Shape: {A_matrix.shape}")

# ---- PHASE 2: Extract Weights for Comparison ----
print("\n⚖️ PHASE 2: Extracting Static Weights (Layer 10 FFN gate_proj) for comparison...")
W_matrix = model.model.layers[10].mlp.gate_proj.weight.data.float().numpy()
print(f"✅ Weights Matrix Shape: {W_matrix.shape}")

del model, outputs, inputs; gc.collect() # Free RAM

# ---- PHASE 3: SVD & Power-Law Analysis ----
print("\n📐 PHASE 3: SVD (Singular Value Decomposition) on BOTH...")

def analyze_spectrum(M, name):
    # SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Variance explained
    var = S**2
    total_var = var.sum()
    cum_var = np.cumsum(var) / total_var
    
    # Power-law slope (log-log fit on top 100 singular values)
    # S[i] ~ i^(-alpha)  => log(S) = -alpha * log(i) + c
    valid_idx = np.arange(1, min(200, len(S)))
    log_i = np.log10(valid_idx)
    log_S = np.log10(S[1:min(200, len(S))] + 1e-9)
    
    A_fit = np.vstack([log_i, np.ones_like(log_i)]).T
    slope, _ = np.linalg.lstsq(A_fit, log_S, rcond=None)[0]
    alpha = -slope
    
    # Metrics
    v10 = cum_var[9] * 100
    v50 = cum_var[49] * 100
    v100 = cum_var[99] * 100
    
    return S, cum_var, alpha, v10, v50, v100

S_act, cum_act, alpha_act, v10_act, v50_act, v100_act = analyze_spectrum(A_matrix, "Activations")
S_wgt, cum_wgt, alpha_wgt, v10_wgt, v50_wgt, v100_wgt = analyze_spectrum(W_matrix, "Weights")

# ---- PHASE 4: THE VERDICT ----
print("\n" + "="*80)
print("📊 THE VERDICT: Behavior vs Hardware")
print("="*80)
print(f"{'Metric':<25}{'Activations (Behavior)':<25}{'Weights (Hardware)':<25}")
print("-"*75)
print(f"{'Power-Law Exponent (α)':<25}{alpha_act:<25.3f}{alpha_wgt:<25.3f}")
print(f"{'Variance in Top 10':<25}{v10_act:<25.2f}%{v10_wgt:<25.2f}%")
print(f"{'Variance in Top 50':<25}{v50_act:<25.2f}%{v50_wgt:<25.2f}%")
print(f"{'Variance in Top 100':<25}{v100_act:<25.2f}%{v100_wgt:<25.2f}%")

print("\n💡 INTERPRETATION:")
if alpha_act > 1.0 and alpha_act > alpha_wgt * 1.5:
    print(f"✅ BREAKTHROUGH! Activations ka α = {alpha_act:.2f} (Weights = {alpha_wgt:.2f})")
    print(f"   Activations HIGHLY STRUCTURED hain! Knowledge compressible hai!")
    print(f"   Top 50 components mein {v50_act:.1f}% knowledge hai.")
    print(f"   Tera Reverse DNA idea ACTIVATIONS pe 100% kaam karega!")
    verdict = "✅ BEHAVIOR IS COMPRESSIBLE!"
    color = 'lightgreen'
elif alpha_act > alpha_wgt:
    print(f"⚠️ PARTIAL. Activations (α={alpha_act:.2f}) thode structured hain weights (α={alpha_wgt:.2f}) se.")
    print(f"   Lekin strong power-law nahi hai. Hybrid approach chahiye hoga.")
    verdict = "⚠️ PARTIALLY COMPRESSIBLE"
    color = 'lightyellow'
else:
    print(f"❌ SHOCKING. Activations bhi white-noise (α={alpha_act:.2f}) hain.")
    print(f"   Knowledge na weights mein structured hai, na activations mein.")
    print(f"   Humein Level 4 (Function/Algorithm extraction) pe jaana padega.")
    verdict = "❌ WHITE NOISE EVERYWHERE"
    color = 'mistyrose'

# ---- PLOTTING ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Singular Value Decay (Log-Log)
axes[0].loglog(S_act, label=f'Activations (α={alpha_act:.2f})', color='#4ECDC4', lw=3)
axes[0].loglog(S_wgt, label=f'Weights (α={alpha_wgt:.2f})', color='#FF6B6B', lw=3, alpha=0.6)
axes[0].set_xlabel('Singular Value Index (log)')
axes[0].set_ylabel('Magnitude (log)')
axes[0].set_title('Spectral Decay (Steeper = More Compressible)')
axes[0].legend(); axes[0].grid(alpha=0.3, which='both')

# Plot 2: Cumulative Variance
axes[1].semilogx(cum_act * 100, label='Activations (Behavior)', color='#4ECDC4', lw=3)
axes[1].semilogx(cum_wgt * 100, label='Weights (Hardware)', color='#FF6B6B', lw=3, alpha=0.6)
axes[1].axhline(95, color='gray', ls='--', alpha=0.5, label='95% Target')
axes[1].set_xlabel('Number of Components (log)')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Knowledge Concentration')
axes[1].legend(); axes[1].grid(alpha=0.3)

# Add Verdict Text to Figure
fig.text(0.5, 0.02, verdict, ha='center', fontsize=20, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1])
os.makedirs('results', exist_ok=True)
plt.savefig('results/optionB_activations_vs_weights.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("🎯 NEXT STEP BASED ON DATA:")
print("="*80)

output
 ================================================================================
🧠 THE MOMENT OF TRUTH: Are Activations Compressible?
================================================================================

🦙 Loading TinyLlama 1.1B (FP16)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100% 608/608 [00:00<00:00, 16.5kB/s]model.safetensors: reconstructing file: 100% 2.20GB / 2.20GB,  112MB/s  model.safetensors: downloading bytes:  1.90GB,  102MB/s  Loading weights: 100% 201/201 [00:03<00:00, 66.56it/s]generation_config.json: 100% 124/124 [00:00<00:00, 10.1kB/s]tokenizer_config.json: 100% 1.29k/1.29k [00:00<00:00, 49.7kB/s]tokenizer.model: reconstructing file: 100%  500kB /  500kB, 49.0kB/s  tokenizer.model: downloading bytes:   346kB, 34.0kB/s  tokenizer.json: 100% 1.84M/1.84M [00:00<00:00, 20.6MB/s]special_tokens_map.json: 100% 551/551 [00:00<00:00, 27.3kB/s]
📝 PHASE 1: Generating Diverse Prompts to extract Activations...
⏳ Running forward pass to capture Hidden States (Layer 10)...
✅ Activations Matrix Shape: (2048, 2048)

⚖️ PHASE 2: Extracting Static Weights (Layer 10 FFN gate_proj) for comparison...
✅ Weights Matrix Shape: (5632, 2048)

📐 PHASE 3: SVD (Singular Value Decomposition) on BOTH...

================================================================================
📊 THE VERDICT: Behavior vs Hardware
================================================================================
Metric                   Activations (Behavior)   Weights (Hardware)       
---------------------------------------------------------------------------
Power-Law Exponent (α)   0.810                    0.150                    
Variance in Top 10       99.39                    %4.08                     %
Variance in Top 50       99.80                    %11.81                    %
Variance in Top 100      99.91                    %18.83                    %

💡 INTERPRETATION:
⚠️ PARTIAL. Activations (α=0.81) thode structured hain weights (α=0.15) se.
   Lekin strong power-law nahi hai. Hybrid approach chahiye hoga.

================================================================================
🎯 NEXT STEP BASED ON DATA:
================================================================================
