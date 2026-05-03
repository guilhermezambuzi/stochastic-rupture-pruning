# Stochastic Rupture Pruning (SRP)

Adaptive inference algorithm for transformer-based neural networks inspired by the Stochastic Rupture (SR) framework of objective wavefunction collapse.

## What is SRP?

In quantum mechanics, the SR framework proposes that wavefunction branches are pruned when local entropy approaches the Bekenstein-Bousso information bound. SRP transposes this principle to neural computation: attention heads are pruned when cumulative informational cost approaches a saturation threshold η.

**The result: significant compute reduction with controlled quality loss.**

## Empirical Results (GPT-2 small)

| η | Heads pruned | Compute reduction | Quality degradation | Ratio |
|---|---|---|---|---|
| 0.90 | 1/12 | 8.3% | 8.0% | 1.04× |
| 0.75 | 3/12 | 25.0% | 13.2% | 1.89× |
| 0.60 | 4/12 | 33.3% | 14.9% | 2.24× |
| 0.50 | 5/12 | 41.7% | 18.6% | 2.24× |
| 0.40 | 7/12 | 58.3% | 51.1% | 1.14× |

**Sweet spot: η = 0.60** — 33% compute reduction with only 15% quality loss (ratio 2.24×).

**Phase transition observed at η = 0.40**: quality degradation jumps from 18.6% to 51.1% with only 2 additional heads pruned. This discontinuous behavior is predicted by the SR framework and mirrors the collapse dynamics of quantum branch pruning.

## How it works

1. Compute efficiency score for each attention head: `E_h = ||V_h|| / C_h`
2. Sort heads by efficiency (descending)
3. Activate heads in order, tracking cumulative saturation χ
4. When χ ≥ η, prune remaining heads irreversibly

## Installation

```bash
pip install torch transformers
```

## Usage

```bash
python srp_teste.py        # basic SRP demo
python srp_perplex.py      # perplexity benchmark with η=0.60
```

## Theoretical Background

SRP is grounded in the Stochastic Rupture (SR) framework (Zambuzi, 2026), which proposes that wavefunction collapse is triggered by local von Neumann entropy approaching the Bekenstein-Bousso information bound. The mapping is formal, not merely metaphorical:

- Quantum branches → attention heads
- von Neumann entropy → cumulative FLOP cost
- Bekenstein-Bousso bound → total layer budget I_max
- Collapse threshold η → pruning threshold η

## Author

Guilherme Zambuzi — Independent Researcher  
In collaboration with Viswak R. Balaji and Samuel Punch (University College Cork)

## License

Apache 2.0
