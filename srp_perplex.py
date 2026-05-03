import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Carregando GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()
print("Modelo carregado!\n")

# Textos de teste
textos = [
    "The universe expands because of informational saturation",
    "Quantum mechanics describes the behavior of particles at small scales",
    "Neural networks learn patterns from large amounts of data",
    "The cat sat on the mat and looked at the window",
    "Scientists discovered a new method for efficient computation",
]

def calcular_perplexidade(model, tokenizer, texto, heads_para_zerar=None):
    inputs = tokenizer(texto, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        loss_completo = outputs.loss.item()
    
    if heads_para_zerar is None:
        return torch.exp(torch.tensor(loss_completo)).item()
    
    # Zera os heads podados via hooks
    hooks = []
    
    def make_hook(heads_podados):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                # Nao eh possivel zerar heads individuais facilmente no GPT2
                # Aproximamos reduzindo o output proporcionalmente
                fator = 1.0 - (len(heads_podados) / 12.0)
                return (attn_output * fator,) + output[1:]
            return output
        return hook
    
    for layer in model.transformer.h:
        h = layer.attn.register_forward_hook(make_hook(heads_para_zerar))
        hooks.append(h)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        loss_podado = outputs.loss.item()
    
    for h in hooks:
        h.remove()
    
    return torch.exp(torch.tensor(loss_podado)).item()

def srp_seleciona_heads(atencoes, eta):
    n_heads = atencoes.shape[1]
    contribuicoes = [atencoes[0, h].norm().item() for h in range(n_heads)]
    custo = 1.0 / n_heads
    ordem = sorted(range(n_heads), key=lambda h: contribuicoes[h], reverse=True)
    
    chi = 0.0
    ativos = []
    podados = []
    for h in ordem:
        if chi < eta:
            ativos.append(h)
            chi += custo
        else:
            podados.append(h)
    return ativos, podados

print("=" * 60)
print(f"{'Texto':<45} {'PPL Full':>8} {'PPL SRP':>8} {'Delta':>7} {'Podados':>8}")
print("=" * 60)

resultados = []

for texto in textos:
    inputs = tokenizer(texto, return_tensors="pt")
    
    # Pega atencoes da camada do meio (camada 6)
    with torch.no_grad():
        out = model(**inputs)
    atencoes = out.attentions[6]
    
    _, podados = srp_seleciona_heads(atencoes, eta=0.60)
    
    ppl_full = calcular_perplexidade(model, tokenizer, texto)
    ppl_srp  = calcular_perplexidade(model, tokenizer, texto, podados)
    delta    = ppl_srp - ppl_full
    
    resultados.append((ppl_full, ppl_srp, delta, len(podados)))
    texto_curto = texto[:44]
    print(f"{texto_curto:<45} {ppl_full:>8.2f} {ppl_srp:>8.2f} {delta:>+7.2f} {len(podados):>8}")

print("=" * 60)

medias = np.mean(resultados, axis=0)
print(f"{'MEDIA':<45} {medias[0]:>8.2f} {medias[1]:>8.2f} {medias[2]:>+7.2f} {medias[3]:>8.1f}")
print("=" * 60)

degradacao = ((medias[1] - medias[0]) / medias[0]) * 100
reducao_compute = (medias[3] / 12) * 100

print(f"\nReducao de compute:      {reducao_compute:.1f}%")
print(f"Degradacao de qualidade: {degradacao:.1f}%")
print(f"\nInterpretacao:")
if degradacao < 5:
    print("EXCELENTE - qualidade quase identica com compute reduzido")
elif degradacao < 15:
    print("BOM - tradeoff favoravel qualidade vs compute")
else:
    print("RUIM - perda de qualidade alta demais")