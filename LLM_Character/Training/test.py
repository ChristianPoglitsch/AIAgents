from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Modell und Tokenizer laden
model = GPT2LMHeadModel.from_pretrained("./my-llm")  # Pfad zu deinem gespeicherten Modell
tokenizer = GPT2Tokenizer.from_pretrained("./my-llm")

# Sicherstellen, dass das Padding-Token korrekt gesetzt ist
tokenizer.pad_token = tokenizer.eos_token

prompt = "Game development of Valkyria Chronicles III"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs.input_ids,
    max_length=50,         # Maximale Laenge der generierten Sequenz
    num_return_sequences=1,  # Anzahl der generierten Sequenzen
    temperature=0.7,        # Kreativitaet (niedrig = konservativer, hoch = kreativer)
    top_k=50,               # Begrenzung auf die Top-k-Wahrscheinlichkeiten
    top_p=0.9,              # Nucleus Sampling (kumulierte Wahrscheinlichkeit)
    do_sample=True          # Sampling aktivieren
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
