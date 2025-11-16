from transformers import AutoTokenizer

models = [
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "microsoft/Phi-3-mini-4k-instruct",
  "mistralai/Mistral-7B-Instruct-v0.2"
]

sentence = "Fine-tuning a small model is both art and science."

for m in models:
    tok = AutoTokenizer.from_pretrained(m)
    tokens = tok.tokenize(sentence)
    print(f"{m}: {len(tokens)} tokens — {tokens}")