 ========================================================================
🧬 MISSION 1.13: NEAREST-NEIGHBOR CONCEPT BANK RESCUE
========================================================================
Loading weights: 100% 290/290 [00:00<00:00, 402.10it/s]✅ LM Head loaded: (151936, 896)
⏳ Precomputing Rank-16 DNA bases...
✅ DNA bases ready (3.3s)
🧠 Building Concept Memory Bank from calibration trajectories...
✅ Concept Bank ready: 55 stored states
⏱️ Bank build time: 31.0s

🔍 UNSEEN PROMPT: 'The capital of Japan is'
Step Context       Original Top-2                  DNA+Bank Rescue                 Match   NN Sim
---------------------------------------------------------------------------------------------------------
0    [PROMPT]      located(28.7%) __(14.0%)Paris(28.7%) ______(12.2%)      🔴 FLIP  1.000
1    'located'     in(45.7%) on(32.0%)is(56.8%) was(12.0%)            🔴 FLIP  0.997
2    'in'          the(44.3%) which(26.4%)is(56.8%) was(12.2%)            🔴 FLIP  0.994
3    'the'         ____(10.0%) ___(7.1%)man(5.9%) (n(1.8%)              🔴 FLIP  0.997
4    '____'        .(44.5%) region(18.3%)…(8.4%) :";(2.9%)               🔴 FLIP  0.998
========================================================================
🏆 VERDICT:
   Agar YES aaye: Concept Bank rescue kaam karta hai for in-distribution/near-calibration prompts.
   Agar FLIP aaye: Bank coverage kam hai, ya DNA state calibration manifold se door hai.
   Ye ab live teacher use nahi kar raha, sirf stored memory use kar raha hai.
⏱️ Time: 37.3s
64