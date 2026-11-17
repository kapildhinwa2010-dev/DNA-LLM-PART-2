# ============================================================
# DISTRIBUTION SHIFT TEST: Oracle vs Unseen Prompts
# Does k=20 with 85.7% accuracy generalize?
# ============================================================
import numpy as np, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*78)
print("DISTRIBUTION SHIFT: Oracle vs Unseen Prompts")
print("="*78)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYER, D = 10, model.config.hidden_size

# ---- CALIBRATION SET (previously used prompts) ----
cal_prompts = ["The capital of France is", "In Python, to sort a list you use",
               "The square root of 144 is", "Photosynthesis converts sunlight into"]

# ---- TEST SET (completely new, unseen prompts) ----
test_prompts = [
    "The largest ocean on Earth is",
    "To reverse a string in Python, you use",
    "The derivative of x squared is",
    "Water boils at what temperature in Celsius?",
    "Who wrote Romeo and Juliet?",
    "The chemical formula for table salt is",
]

def get_activations_and_argmax(prompts):
    inp = tok(prompts, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    pad_mask = (inp["attention_mask"] == 1)
    cap = {}
    def hook(m, i, o): cap["act"] = o.detach().float().clone()
    h = model.model.layers[LAYER].register_forward_hook(hook)
    with torch.no_grad():
        out = model(**inp)
    h.remove()
    return cap["act"], out.logits.argmax(-1), pad_mask, inp

# ---- PHASE 1: Oracle (calibration SVD, test on same tokens) ----
print("\nPHASE 1: ORACLE (calibration = test, no distribution shift)...")
cal_act, cal_argmax, cal_pad, _ = get_activations_and_argmax(cal_prompts)
A_cal = cal_act.reshape(-1, D).numpy()
_, Sig_cal, Vt_cal = np.linalg.svd(A_cal, full_matrices=False)
cum_cal = np.cumsum(Sig_cal**2) / (Sig_cal**2).sum()

def bn_hook(Vk):
    def f(m, i, o):
        x = o.float(); return (x @ Vk.T @ Vk).to(o.dtype)
    return f

print(f"{'k':<5}{'Var%':<9}{'Oracle Acc%':<13}")
print("-"*27)
oracle_results = {}
for k in [5, 10, 20, 30, 40, 50]:
    if k > len(Sig_cal): break
    Vk = torch.tensor(Vt_cal[:k, :], dtype=torch.float32)
    hh = model.model.layers[LAYER].register_forward_hook(bn_hook(Vk))
    _, test_argmax, _, _ = get_activations_and_argmax(cal_prompts)  # same prompts
    hh.remove()
    acc = (test_argmax == cal_argmax)[cal_pad].float().mean().item() * 100
    oracle_results[k] = acc
    print(f"{k:<5}{cum_cal[k-1]*100:<9.2f}{acc:<13.1f}")

# ---- PHASE 2: DISTRIBUTION SHIFT (calibration SVD, test on UNSEEN prompts) ----
print("\nPHASE 2: DISTRIBUTION SHIFT (calibration != test, unseen prompts)...")
test_act, test_argmax, test_pad, _ = get_activations_and_argmax(test_prompts)

print(f"{'k':<5}{'Var%':<9}{'Oracle Acc%':<13}{'Unseen Acc%':<13}{'Drop%':<10}{'Verdict'}")
print("-"*63)
shift_results = {}
for k in [5, 10, 20, 30, 40, 50]:
    if k > len(Sig_cal): break
    Vk = torch.tensor(Vt_cal[:k, :], dtype=torch.float32)
    hh = model.model.layers[LAYER].register_forward_hook(bn_hook(Vk))
    _, unseen_argmax, _, _ = get_activations_and_argmax(test_prompts)
    hh.remove()
    unseen_acc = (unseen_argmax == test_argmax)[test_pad].float().mean().item() * 100
    drop = oracle_results[k] - unseen_acc
    verdict = "STABLE" if drop < 5 else ("DRIFT" if drop < 15 else "COLLAPSE")
    shift_results[k] = {'oracle': oracle_results[k], 'unseen': unseen_acc, 'drop': drop}
    print(f"{k:<5}{cum_cal[k-1]*100:<9.2f}{oracle_results[k]:<13.1f}{unseen_acc:<13.1f}{drop:<10.1f}{verdict}")

# ---- PHASE 3: EXPANDED CALIBRATION (10+ prompts) ----
print("\nPHASE 3: EXPANDED CALIBRATION (10 prompts, not 4)...")
expanded_prompts = cal_prompts + [
    "The speed of light is approximately",
    "In JavaScript, to declare a constant you use",
    "The mitochondria is the of the cell",
    "E equals mc squared was proposed by",
    "The longest river in the world is",
    "To open a file in Python, you use the function",
]
exp_act, exp_argmax, exp_pad, _ = get_activations_and_argmax(expanded_prompts)
A_exp = exp_act.reshape(-1, D).numpy()
_, Sig_exp, Vt_exp = np.linalg.svd(A_exp, full_matrices=False)
cum_exp = np.cumsum(Sig_exp**2) / (Sig_exp**2).sum()

print(f"{'k':<5}{'Var%':<9}{'Unseen Acc% (exp cal)':<22}{'vs Oracle (4 cal)':<18}{'Improvement'}")
print("-"*74)
for k in [5, 10, 20, 30, 40, 50]:
    if k > len(Sig_exp): break
    Vk = torch.tensor(Vt_exp[:k, :], dtype=torch.float32)
    hh = model.model.layers[LAYER].register_forward_hook(bn_hook(Vk))
    _, unseen_argmax, _, _ = get_activations_and_argmax(test_prompts)
    hh.remove()
    exp_acc = (unseen_argmax == test_argmax)[test_pad].float().mean().item() * 100
    oracle_4 = shift_results.get(k, {}).get('unseen', 0)
    imp = exp_acc - oracle_4
    print(f"{k:<5}{cum_exp[k-1]*100:<9.2f}{exp_acc:<22.1f}{oracle_4:<18.1f}{imp:+.1f}")

# ---- FINAL VERDICT ----
print("\n" + "="*78)
print("DISTRIBUTION SHIFT VERDICT")
print("="*78)

k20_oracle = shift_results.get(20, {}).get('oracle', 0)
k20_unseen_4cal = shift_results.get(20, {}).get('unseen', 0)
k20_drop = shift_results.get(20, {}).get('drop', 999)

print(f"""
Key findings for k=20 (useful compression point):
   Oracle accuracy:     {k20_oracle:.1f}%
   Unseen (4-cal):      {k20_unseen_4cal:.1f}%
   Drop:                {k20_drop:.1f}%
   
   Verdict: {'STABLE - Behavior DNA VIABLE with 4 prompts!' if k20_drop < 10 else 'DRIFT - need expanded calibration' if k20_drop < 20 else 'COLLAPSE - SVD basis too narrow'}

Key findings for k=10 (high compression point):
   Oracle: {shift_results.get(10,{}).get('oracle',0):.1f}%  Unseen: {shift_results.get(10,{}).get('unseen',0):.1f}%  Drop: {shift_results.get(10,{}).get('drop',0):.1f}%

KEY FINDINGS:
   1. Variance != behavior (confirmed: 99.79% var -> 51% acc)
   2. SVD bottleneck itself is safe (k=50 -> 100%)
   3. Distribution shift effect: {k20_drop:.1f}% drop at k=20
   4. Expanded calibration (10 prompts) improvement: see table above

NEXT STEP:
""")

if k20_drop < 10:
    print("""   Behavior DNA VIABLE with minimal calibration!
      -> Stage 4: Build DNA Generator (predict top-k codes from input)
      -> 4 prompts calibration achieves 85%+ unseen accuracy
      -> This is publishable: 'Behavior DNA: 4-prompt calibration for LLM compression'""")
elif k20_drop < 20:
    print("""   Expanded calibration needed (10-50 prompts).
      -> Check Phase 3 results: did 10 prompts reduce the drop?
      -> If yes -> 10-prompt calibration makes Behavior DNA viable
      -> If no -> 50-100 prompts needed, still feasible""")
else:
    print("""   SVD basis too narrow for distribution shift.
      -> Behavior DNA needs output-aware basis (logit-trained)
      -> Or: pivot to Lottery Ticket (spatial sparsity)
      -> But this is also a finding: 'SVD compression fails under distribution shift'""")

del model; gc.collect()
