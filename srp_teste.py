import torch
from transformers import GPT2Model, GPT2Tokenizer

# Carrega GPT-2 small (124M parametros)
print("Carregando GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
model.eval()
print("Modelo carregado!")

# Texto de teste
texto = "The universe expands because of informational saturation"
inputs = tokenizer(texto, return_tensors="pt")

# Roda o modelo
print("\nRodando modelo completo...")
with torch.no_grad():
    outputs = model(**inputs)

# Pega os pesos de atencao da primeira camada
atencoes = outputs.attentions[0]  # camada 0
print(f"Shape das atencoes: {atencoes.shape}")
print(f"Numero de heads: {atencoes.shape[1]}")

# Calcula contribuicao de cada head (norma L2)
contribuicoes = []
for h in range(atencoes.shape[1]):
    norma = atencoes[0, h].norm().item()
    contribuicoes.append(norma)
    print(f"Head {h:02d}: contribuicao = {norma:.4f}")

# SRP: ordena por eficiencia e prune quando chi >= eta
eta = 0.60
custo_por_head = 1.0 / atencoes.shape[1]  # custo uniforme por enquanto
I_max = custo_por_head * atencoes.shape[1]

ordem = sorted(range(len(contribuicoes)), 
               key=lambda h: contribuicoes[h], reverse=True)

chi = 0.0
heads_ativos = []
heads_podados = []

print(f"\n--- SRP com eta={eta} ---")
for h in ordem:
    if chi < eta:
        heads_ativos.append(h)
        chi += custo_por_head
        print(f"Head {h:02d} ATIVO   | chi={chi:.2f}")
    else:
        heads_podados.append(h)
        print(f"Head {h:02d} PODADO  | chi={chi:.2f}")

print(f"\nHeads ativos:  {len(heads_ativos)}/{atencoes.shape[1]}")
print(f"Heads podados: {len(heads_podados)}/{atencoes.shape[1]}")
print(f"Reducao de compute: {len(heads_podados)/atencoes.shape[1]*100:.1f}%")