# Qualitative Results: SRP Improves Output Quality in Deep Models

Date: 2026-05-03
Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (28 layers, GQA)

## Key Finding

At η=0.75, SRP produces **better qualitative outputs** than the full model (η=1.0).
The full model enters repetition loops that SRP eliminates.

## Evidence

### Q1: "What is photosynthesis? Explain simply."

**η=1.0 (full model):** Enters infinite loop — "Also, what is the role of chlorophyll... Also, what is the role of water... Also, what is the role of the sun..." — unusable output.

**η=0.75 (SRP):** Correct, concise answer with chemical equation (6CO2 + H2O → C6H12O6 + O2). Functional output.

**η=0.60:** Starts correctly but enters "chloroplasts" repetition loop. Degraded.

---

### Q2: "What causes inflation in an economy?"

**η=1.0:** Good but generic answer. No structure.

**η=0.75 (SRP):** Best answer — structured, uses bold headers, more precise. Covers supply/demand, monetary policy, globalization.

**η=0.60:** Drifts into Fisher equation loop. Degraded.

---

### Q3: "How do neural networks learn?"

**η=1.0:** Rambles without answering directly.

**η=0.75 (SRP):** Best answer — clear step-by-step reasoning, explains backpropagation naturally.

**η=0.60:** Starts well, fragments at the end.

---

## Interpretation

DeepSeek-R1-Distill-Qwen-1.5B (28 layers) has **excess attentional redundancy**.
SRP at η=0.75 prunes this redundancy, producing:

1. **No repetition loops** (full model fails here)
2. **More structured responses**
3. **Better factual focus**

This suggests SRP is not only a compute efficiency mechanism but potentially
a **quality improvement mechanism** for deep modern architectures.

## Hypothesis

Modern deep transformers may be over-attended — carrying more attentional
capacity than needed for factual generation. SRP's saturation-based pruning
selectively removes this excess, acting as a form of informational regularization.

This is consistent with the SR framework prediction: systems operating
**below saturation** are more coherent than saturated systems.

## Next Steps

- [ ] Test on larger dataset (WikiText-103)
- [ ] Test on Llama-3, Mistral, Qwen-2
- [ ] Compare with dropout and other regularization methods
- [ ] Investigate if η=0.75 sweet spot generalizes across model families
