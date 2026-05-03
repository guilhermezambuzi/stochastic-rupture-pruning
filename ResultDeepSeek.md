# DeepSeek-R1-Distill-Qwen-1.5B Results

Date: 2026-05-03
Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Architecture: Grouped Query Attention (12 Q heads, 2 KV heads, 28 layers)
Test: SRP via proportional scaling on 10 sentences

## Results

| η | DeepSeek PPL | Delta | GPT-2 small (reference) |
|---|---|---|---|
| baseline | 646.69 | — | 356.60 |
| 0.90 | 593.92 | **−8.2%** | +8.0% |
| 0.75 | 548.50 | **−15.2%** | +13.2% |
| 0.60 | 696.08 | +7.6% | +14.9% |
| 0.50 | 1084.00 | +67.6% | +18.6% |
| 0.40 | 3743.25 | +478.8% | +51.1% |

## Key findings

1. **Modern GQA models are more resilient to mild SRP**: at η=0.75 and η=0.90, perplexity actually decreases, suggesting that DeepSeek has redundant attention capacity that SRP can prune beneficially.

2. **Phase transition is sharper**: while GPT-2 transitions from 18.6% to 51.1% degradation (×2.7), DeepSeek transitions from 67.6% to 478.8% (×7). The discontinuous behavior predicted by the SR framework is more pronounced in deeper modern architectures.

3. **Sweet spot shifted**: optimal operating point appears at η≈0.75-0.80 for DeepSeek (negative delta), versus η=0.60 for GPT-2.

## Caveats

- These results use proportional output scaling, not surgical head pruning. Real pruning is expected to produce higher degradation.
- Tested on 10 sentences only. Larger-scale evaluation on WikiText-103 is future work.
- GQA architecture means independent query head pruning is non-trivial; full SRP-GQA implementation is pending.

## Reproducibility

Code and Colab notebook: see `srp_perplex.py` and adapt model loading for DeepSeek via `AutoModelForCausalLM`.
