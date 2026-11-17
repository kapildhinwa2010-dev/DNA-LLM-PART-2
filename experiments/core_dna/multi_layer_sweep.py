# ============================================================
# 🧬 MULTI-LAYER DNA SWEEP: 1 vs 7 vs 24 layers (cascade test)
# ============================================================
import numpy as np, torch, gc, time
from transformers import AutoModelForCausalLM, AutoTokenizer

t0=time.time(); print("="*72); print("🧬 MULTI-LAYER DNA SWEEP (cascade fix test)"); print("="*72)
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",dtype=torch.float32,low_cpu_mem_usage=True)
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); tok.padding_side='left'
if tok.pad_token is None: tok.pad_token=tok.eos_token
model.eval(); NL=model.config.num_hidden_layers; D=model.config.hidden_size; ML=24

cal=["The capital of France is","The capital of Germany is","The capital of Japan is",
"The chemical symbol for gold is","The chemical symbol for oxygen is","The square of 5 is",
"The square of 10 is","France is in Europe","Japan is in Asia","The atomic number of carbon is",
"The atomic number of iron is","The capital of Brazil is","The chemical symbol for sodium is",
"The square of 7 is","Brazil is in South America","The atomic number of helium is",
"The capital of Egypt is","The chemical symbol for copper is","The square of 12 is",
"Egypt is in Africa","The atomic number of neon is","The capital of Canada is",
"The chemical symbol for zinc is","The square of 8 is","Canada is in North America",
"The atomic number of argon is","The capital of India is","The chemical symbol for silver is",
"The square of 15 is","India is in Asia"]
test=["The capital of Mexico is","The chemical symbol for lead is","The square of 17 is",
"Mexico is in North America","The atomic number of calcium is","The capital of Sweden is",
"The chemical symbol for tin is","The square of 18 is","Sweden is in Europe",
"The atomic number of sulfur is","The capital of Poland is","The chemical symbol for lithium is",
"The square of 19 is","Poland is in Europe","The atomic number of magnesium is"]

def extract(prompts, layers):
    """Extract (in,out) deltas for all layers + baseline argmax in one forward"""
    inp=tok(prompts,return_tensors="pt",padding="max_length",max_length=ML,truncation=True)
    pm=inp["attention_mask"]==1; box={}
    hs=[]
    for l in layers:
        def mk(l):
            def hook(m,ii,o):
                box[l]=(ii[0].detach().float(), o.detach().float()); return o
            return hook
        hs.append(model.model.layers[l].register_forward_hook(mk(l)))
    with torch.no_grad(): out=model(**inp)
    for h in hs: h.remove()
    return box, pm, out.logits.argmax(-1)

print("📐 Extracting ALL 24 layers (cal + test), single forward each...")
ALL=list(range(NL))
bc,pm_c,_ = extract(cal, ALL)
bt,pm_t,base_am = extract(test, ALL)
print(f"   done ({time.time()-t0:.0f}s) | cal tok={pm_c.sum().item()} test tok={pm_t.sum().item()}")

# ---- Per-layer basis (delta SVD) ----
BAS={}
for l in ALL:
    hin,hout=bc[l]; Dc=(hout-hin)[pm_c].numpy(); mu=Dc.mean(0)
    _,S,Vt=np.linalg.svd(Dc-mu,full_matrices=False)
    BAS[l]=(Vt,mu,S)
def Rmat(l, k):
    """oracle replay matrix: projected true delta (B,L,D)"""
    Vt,mu,S=BAS[l]; k=min(k,Vt.shape[0])
    hin,hout=bt[l]; Dt=(hout-hin)[pm_t].numpy()
    R=torch.zeros_like(hout); R[pm_t]=torch.tensor(((Dt-mu)@Vt[:k].T)@Vt[:k]+mu,dtype=torch.float32)
    return R

def eval_layers(layers, k):
    Rs={l:Rmat(l,k) for l in layers}
    hs=[]
    for l in layers:
        def mk(l):
            def hook(m,ii,o): return (ii[0]+Rs[l]).to(o.dtype)
            return hook
        hs.append(model.model.layers[l].register_forward_hook(mk(l)))
    inp_t=tok(test,return_tensors="pt",padding="max_length",max_length=ML,truncation=True)
    with torch.no_grad(): am=model(**inp_t).logits.argmax(-1)
    for h in hs: h.remove()
    return ((am==base_am).float().mean().item())*100

MID7=[8,9,10,11,12,13,14]
strats=[
 ("1 layer  (L11) r16   [control]", [11], 16),
 ("4 layers (8-11) r16",            [8,9,10,11], 16),
 ("4 layers (11-14) r16",           [11,12,13,14], 16),
 ("7 layers (8-14) r16",            MID7, 16),
 ("7 layers (8-14) r32",            MID7, 32),
 ("7 layers (8-14) r64",            MID7, 64),
 ("ALL 24 layers r16",              ALL, 16),
 ("ALL 24 layers r32",              ALL, 32),
]
print(f"\n{'Strategy':<32}{'Layers':<8}{'Unseen Acc%':<12}")
print("-"*54)
res=[]
for name,layers,k in strats:
    acc=eval_layers(layers,k); res.append((name,len(layers),acc))
    print(f"{name:<32}{len(layers):<8}{acc:<12.1f}")

ctrl=[a for n,l,a in res if n.startswith('1 layer')][0]
m7=[a for n,l,a in res if '7 layers' in n and 'r16' in n][0]
a24=[a for n,l,a in res if n.startswith('ALL') and 'r16' in n][0]
print("\n"+"="*72)
print(f"📊 Cascade verdict: 1-layer={ctrl:.1f}% | 7-layer={m7:.1f}% | 24-layer={a24:.1f}%")
if m7>ctrl+5: print("✅ MULTI-LAYER cascade works — knowledge is jointly compressible!")
elif a24>ctrl+5: print("✅ FULL-MODEL DNA viable — compress all layers together!")
elif m7>ctrl or a24>ctrl: print("⚠️ Some improvement — partial cascade fix.")
else: print("❌ Even multi-layer cannot fix cascade — knowledge is layer-coupled, not representation-coupled.")
print(f"⏱️ {time.time()-t0:.0f}s"); print("="*72)
del model; gc.collect()