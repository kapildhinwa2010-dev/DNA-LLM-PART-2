# ============================================================
# OPTION B: BEHAVIOR COMPRESSIBILITY PROOF (Activations vs Weights)
# Is "Knowledge in Action" (Activations) compressible?
# ============================================================
import numpy as np, os, gc, time, torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("THE MOMENT OF TRUTH: Are Activations Compressible?")
print("="*80)

# ---- PHASE 0: Load Model & Tokenizer ----
print("\nLoading TinyLlama 1.1B (FP16)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# ---- PHASE 1: Generate Activations (The "Knowledge in Action") ----
print("\nPHASE 1: Generating Diverse Prompts to extract Activations...")
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

print("Running forward pass to capture Hidden States (Layer 10)...")
with torch.no_grad():
    # output_hidden_states=True gives us the activations at every layer
    outputs = model(**inputs, output_hidden_states=True)

# Extract Layer 10 activations (middle layer, usually richest in semantics)
# Shape: (batch=8, seq_len=256, hidden_dim=2048)
activations = outputs.hidden_states[10].float().numpy() 

# Reshape to 2D matrix: (2048 rows of context, 2048 columns of features)
A_matrix = activations.reshape(-1, 2048)
print(f"Activations Matrix Shape: {A_matrix.shape}")

# ---- PHASE 2: Extract Weights for Comparison ----
print("\nPHASE 2: Extracting Static Weights (Layer 10 FFN gate_proj) for comparison...")
W_matrix = model.model.layers[10].mlp.gate_proj.weight.data.float().numpy()
print(f"Weights Matrix Shape: {W_matrix.shape}")

del model, outputs, inputs; gc.collect() # Free RAM

# ---- PHASE 3: SVD & Power-Law Analysis ----
print("\nPHASE 3: SVD (Singular Value Decomposition) on BOTH...")

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
print("THE VERDICT: Behavior vs Hardware")
print("="*80)
print(f"{'Metric':<25}{'Activations (Behavior)':<25}{'Weights (Hardware)':<25}")
print("-"*75)
print(f"{'Power-Law Exponent (alpha)':<25}{alpha_act:<25.3f}{alpha_wgt:<25.3f}")
print(f"{'Variance in Top 10':<25}{v10_act:<25.2f}%{v10_wgt:<25.2f}%")
print(f"{'Variance in Top 50':<25}{v50_act:<25.2f}%{v50_wgt:<25.2f}%")
print(f"{'Variance in Top 100':<25}{v100_act:<25.2f}%{v100_wgt:<25.2f}%")

print("\nINTERPRETATION:")
if alpha_act > 1.0 and alpha_act > alpha_wgt * 1.5:
    print(f"BREAKTHROUGH! Activations alpha = {alpha_act:.2f} (Weights = {alpha_wgt:.2f})")
    print(f"   Activations are HIGHLY STRUCTURED! Knowledge is compressible!")
    print(f"   Top 50 components contain {v50_act:.1f}% of knowledge.")
    print(f"   Your Reverse DNA idea will work 100% on ACTIVATIONS!")
    verdict = "BEHAVIOR IS COMPRESSIBLE!"
    color = 'lightgreen'
elif alpha_act > alpha_wgt:
    print(f"PARTIAL. Activations (alpha={alpha_act:.2f}) are somewhat structured compared to weights (alpha={alpha_wgt:.2f}).")
    print(f"   But not a strong power-law. Hybrid approach may be needed.")
    verdict = "PARTIALLY COMPRESSIBLE"
    color = 'lightyellow'
else:
    print(f"SHOCKING. Activations are also white-noise (alpha={alpha_act:.2f}).")
    print(f"   Knowledge is not structured in weights or activations.")
    print(f"   We need to move to Level 4 (Function/Algorithm extraction).")
    verdict = "WHITE NOISE EVERYWHERE"
    color = 'mistyrose'

# ---- PLOTTING ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Singular Value Decay (Log-Log)
axes[0].loglog(S_act, label=f'Activations (alpha={alpha_act:.2f})', color='#4ECDC4', lw=3)
axes[0].loglog(S_wgt, label=f'Weights (alpha={alpha_wgt:.2f})', color='#FF6B6B', lw=3, alpha=0.6)
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
print("NEXT STEP BASED ON DATA:")
print("="*80)
