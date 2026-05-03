"""
SRP applied to DeepSeek-R1-Distill-Qwen-1.5B
Cross-architecture validation on Grouped Query Attention (GQA)

Usage:
    pip install torch transformers accelerate
    python srp_deepseek.py

Note: Requires GPU with at least 4GB VRAM. Tested on Tesla T4 (Google Colab).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calcular_perplexidade(model, tokenizer, texto, device=DEVICE):
    inputs = tokenizer(texto, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def aplicar_srp_scaling(model, eta=0.6):
    """
    Apply SRP via forward hooks that scale attention output proportionally.
    This is an approximation valid for cross-architecture comparison.
    For surgical head pruning, see srp_perplex.py (GPT-2 specific).
    """
    fator = eta
    hooks = []
    
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (output[0] * fator,) + output[1:]
            return output * fator
        return hook
    
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_hook(make_hook())
        hooks.append(h)
    
    return hooks


def remover_hooks(hooks):
    for h in hooks:
        h.remove()


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()
    
    print(f"\nArchitecture:")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Q heads: {model.config.num_attention_heads}")
    print(f"  KV heads: {model.config.num_key_value_heads}")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    textos = [
        "The universe expands because of informational saturation",
        "Quantum mechanics describes the behavior of particles at small scales",
        "Neural networks learn patterns from large amounts of data",
        "The cat sat on the mat and looked at the window",
        "Scientists discovered a new method for efficient computation",
        "Climate change affects global weather patterns significantly",
        "The economy grew faster than analysts predicted last quarter",
        "Artificial intelligence transforms many industries today",
        "Music has the power to evoke deep emotions in listeners",
        "Children learn language through exposure and practice",
    ]
    
    # Baseline
    print("\n--- Baseline ---")
    ppl_baseline = [calcular_perplexidade(model, tokenizer, t) for t in textos]
    ppl_base_mean = sum(ppl_baseline) / len(ppl_baseline)
    print(f"Mean perplexity: {ppl_base_mean:.2f}")
    
    # SRP curve
    print("\n--- SRP via proportional scaling ---")
    print(f"{'eta':>6} | {'PPL':>10} | {'Delta':>8}")
    print("-" * 35)
    
    for eta in [0.90, 0.75, 0.60, 0.50, 0.40]:
        hooks = aplicar_srp_scaling(model, eta=eta)
        ppl = [calcular_perplexidade(model, tokenizer, t) for t in textos]
        ppl_mean = sum(ppl) / len(ppl)
        remover_hooks(hooks)
        
        delta = ((ppl_mean - ppl_base_mean) / ppl_base_mean) * 100
        print(f"{eta:>6.2f} | {ppl_mean:>10.2f} | {delta:>+7.1f}%")


if __name__ == "__main__":
    main()
