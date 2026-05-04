# Less Attention, Better AI: Stochastic Rupture Pruning (SRP)

Adaptive inference algorithm for transformer-based neural networks inspired by the Stochastic Rupture (SR) framework of objective wavefunction collapse.

**Key finding: pruning attention heads at η=0.75 improves output quality by up to 25.4% in modern LLMs and eliminates repetition loops.**

## What is SRP?

In quantum mechanics, the SR framework proposes that wavefunction branches are pruned when local entropy approaches the Bekenstein-Bousso information bound. SRP transposes this principle to neural computation: attention heads are pruned when cumulative informational cost approaches a saturation threshold η.

The result: significant compute reduction and, counterintuitively, quality improvement in modern deep transformers.

## Cross-Architecture Results

### Perplexity delta vs full model

| η | GPT-2 small | DeepSeek-1.5B | Qwen2.5-1.5B | Mistral-7B |
|---|---|---|---|---|
| 0.90 | +8.0% | −8.2% | −12.6% | −0.9% |
| **0.75** | +13.2% | **−15.2%** | **−25.4%** | **−1.5%** |
| 0.60 | +14.9% | +7.6% | +235.3% | +0.7% |
| 0.50 | +18.6% | +67.6% | +34.3% | +7.9% |
| 0.40 | +51.1% | +478.8% | +3865.2% | +24.4% |

Negative delta = quality **improves** over full model. η=0.75 is optimal across all modern architectures.

---

### Architecture summary

| Model | Parameters | Layers | Attention | Sweet spot | Max improvement |
|---|---|---|---|---|---|
| GPT-2 small | 124M | 12 | MHA | η=0.60 | degrades |
| DeepSeek-R1-1.5B | 1.5B | 28 | GQA (2 KV) | η=0.75 | −15.2% |
| Qwen2.5-1.5B | 1.5B | 28 | GQA (2 KV) | η=0.75 | −25.4% |
| Mistral-7B | 7B | 32 | GQA (8 KV) | η=0.75 | −1.5% |

---

### Qualitative evidence (DeepSeek-1.5B, η=0.75 vs η=1.0)

**Q: "What is photosynthesis? Explain simply."**

| η=1.0 (full model) | η=0.75 (SRP) |
|---|---|
| Infinite loop: "Also, what is the role of chlorophyll... Also, what is the role of water..." — unusable | Correct answer with equation: 6CO2 + H2O → C6H12O6 + O2 |

**Q: "What causes inflation?"**

| η=1.0 | η=0.75 (SRP) |
|---|---|
| Generic unstructured answer | Structured answer with headers |

**Q: "How do neural networks learn?"**

| η=1.0 | η=0.75 (SRP) |
|---|---|
| Rambles without answering | Clear step-by-step reasoning explaining backpropagation |

---

### Phase transition

Sharp discontinuity observed at η=0.40 across all architectures. Severity scales with depth.

| Model | Layers | Jump at η=0.40 |
|---|---|---|
| GPT-2 small | 12 | 2.7x |
| DeepSeek-1.5B | 28 | 7.1x |
| Qwen2.5-1.5B | 28 | 113x |
| Mistral-7B | 32 | 3.1x |

---

## How it works

1. Compute efficiency score for each attention head: `E_h = ||V_h|| / C_h`
2. Sort heads by efficiency (descending)
3. Activate heads in order, tracking cumulative saturation χ
4. When χ ≥ η, prune remaining heads irreversibly

## Theoretical Background

SRP is grounded in the Stochastic Rupture (SR) framework (Zambuzi, 2026), which proposes that wavefunction collapse is triggered by local von Neumann entropy approaching the Bekenstein-Bousso information bound.

| SR framework | SRP |
|---|---|
| Quantum branches | Attention heads |
| von Neumann entropy | Cumulative FLOP cost |
| Bekenstein-Bousso bound | Layer budget I_max |
| Collapse threshold η | Pruning threshold η |
| Branch pruning | Head deactivation |

**Key prediction confirmed:** Systems operating below informational saturation are more coherent than saturated systems. Modern deep transformers appear to be over-attended — SRP prunes this excess, acting as informational regularization.

## Installation

```bash
pip install torch transformers accelerate
```

## Usage

```bash
# Basic SRP demo (GPT-2 small)
python srp_teste.py

# Perplexity benchmark with η sweep (GPT-2 small)
python srp_perplex.py

# Cross-architecture test (DeepSeek-1.5B, requires GPU)
python srp_deepseek.py
```

## Files

| File | Description |
|---|---|
| `srp_teste.py` | Basic SRP proof of concept on GPT-2 small |
| `srp_perplex.py` | Perplexity benchmark, η sweep |
| `srp_deepseek.py` | Cross-architecture validation on DeepSeek-1.5B |
| `deepseek_results.md` | Quantitative results on DeepSeek |
| `qualitative_results.md` | Qualitative output comparison |

## Paper

Zambuzi, G. (2026). *Less Attention, Better AI: Stochastic Rupture Pruning Improves LLM Quality Across Architectures*. Zenodo. https://doi.org/10.5281/zenodo.20027724


## Acknowledgements

Thanks to Viswak R. Balaji and Samuel Punch (University College Cork) for circuit-level Qiskit simulations of the SR framework (arXiv:2508.10590).

## Author

**Guilherme Zambuzi** — Independent Researcher
gzambuzi@gmail.com

## License

Business Source License 1.1 (BSL)
Non-commercial and academic use: unrestricted
Commercial use: contact gzambuzi@gmail.com
