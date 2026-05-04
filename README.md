Stochastic Rupture Pruning (SRP)
Adaptive inference algorithm for transformer-based neural networks inspired by the Stochastic Rupture (SR) framework of objective wavefunction collapse.
Key Results
SRP not only reduces compute — it improves output quality in deep modern models.
At η=0.75, DeepSeek-R1-Distill-Qwen-1.5B produces better, more structured responses than the full model, and eliminates repetition loops that appear at η=1.0.
What is SRP?
In quantum mechanics, the SR framework proposes that wavefunction branches are pruned when local entropy approaches the Bekenstein-Bousso information bound. SRP transposes this principle to neural computation: attention heads are pruned when cumulative informational cost approaches a saturation threshold η.
The result: significant compute reduction with controlled quality loss — and in deep modern models, quality improvement.
Empirical Results
GPT-2 small (12 layers, 124M parameters)
ηHeads prunedCompute reductionQuality degradationRatio0.901/128.3%8.0%1.04×0.753/1225.0%13.2%1.89×0.604/1233.3%14.9%2.24×0.505/1241.7%18.6%2.24×0.407/1258.3%51.1%1.14×
Sweet spot: η = 0.60 — 33% compute reduction with only 15% quality loss (ratio 2.24×).
Phase transition at η = 0.40: quality degradation jumps from 18.6% to 51.1% with only 2 additional heads pruned. This discontinuous behavior is predicted by the SR framework and mirrors the collapse dynamics of quantum branch pruning.

DeepSeek-R1-Distill-Qwen-1.5B (28 layers, GQA, 1.5B parameters)
ηPPL deltavs GPT-2Interpretation0.90−8.2%+8.0%Quality improves0.75−15.2%+13.2%Quality improves0.60+7.6%+14.9%Better than GPT-20.50+67.6%+18.6%Degraded0.40+478.8%+51.1%Collapse
Sweet spot: η = 0.75 for deep models.
Modern deep transformers appear to carry excess attentional redundancy. SRP at η=0.75 prunes this redundancy, acting as informational regularization.

Qualitative Evidence (DeepSeek, η=0.75 vs η=1.0)
Q: "What is photosynthesis? Explain simply."
η=1.0 (full model)η=0.75 (SRP)Enters infinite loop: "Also, what is the role of chlorophyll... Also, what is the role of water..." — unusableCorrect concise answer with chemical equation: 6CO₂ + H₂O → C₆H₁₂O₆ + O₂
Q: "What causes inflation?"
η=1.0η=0.75 (SRP)Generic unstructured answerStructured answer with headers covering supply/demand, monetary policy, globalization
Q: "How do neural networks learn?"
η=1.0η=0.75 (SRP)Rambles without answeringClear step-by-step reasoning explaining backpropagation

How it works

Compute efficiency score for each attention head: E_h = ||V_h|| / C_h
Sort heads by efficiency (descending)
Activate heads in order, tracking cumulative saturation χ
When χ ≥ η, prune remaining heads irreversibly

Algorithm
Input: attention heads H, budget factor β, threshold η
1. E_h ← ||V_h|| / C_h  for all h
2. π ← argsort(E, descending)
3. I_max ← β · Σ C_h
4. χ ← 0; A ← ∅
5. while χ < η and heads remain:
     activate next head h_k in π
     A ← A ∪ {h_k}
     χ ← χ + C_hk / I_max
6. prune all heads not in A
Theoretical Background
SRP is grounded in the Stochastic Rupture (SR) framework (Zambuzi, 2026), which proposes that wavefunction collapse is triggered by local von Neumann entropy approaching the Bekenstein-Bousso information bound. The mapping is formal, not merely metaphorical:
SR frameworkSRPQuantum branchesAttention headsvon Neumann entropyCumulative FLOP costBekenstein-Bousso boundLayer budget I_maxCollapse threshold ηPruning threshold ηBranch pruningHead deactivation
Key prediction confirmed: Systems operating below informational saturation are more coherent than saturated systems. This explains why η=0.75 improves quality in deep models — they were over-saturated.
Installation
bashpip install torch transformers accelerate
Usage
bash# Basic SRP demo (GPT-2 small)
python srp_teste.py

# Perplexity benchmark with η sweep (GPT-2 small)
python srp_perplex.py

# Cross-architecture test (DeepSeek-1.5B, requires GPU)
python srp_deepseek.py
Files
FileDescriptionsrp_teste.pyBasic SRP proof of concept on GPT-2 smallsrp_perplex.pyPerplexity benchmark, η=0.60 sweet spotsrp_deepseek.pyCross-architecture validation on DeepSeek-1.5Bdeepseek_results.mdQuantitative results on DeepSeekqualitative_results.mdQualitative output comparison
Paper
Zambuzi, G. (2026). Stochastic Rupture Pruning: Empirical Validation of Informational Saturation as an Adaptive Compute Bound for Transformer Inference. Zenodo. (DOI pending)
Acknowledgements
Thanks to Viswak R. Balaji and Samuel Punch (University College Cork) for circuit-level Qiskit simulations of the SR framework (arXiv:2508.10590).
Author
Guilherme Zambuzi — Independent Researcher
gzambuzi@gmail.com
License
Apache 2.0
