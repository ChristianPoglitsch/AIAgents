from transformers import GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=256,
    n_layer=4,
    n_head=4
)
model = GPT2LMHeadModel(config)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
#dataset = load_dataset("path/to/your/data")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",          # Speichere das Modell hier
    evaluation_strategy="epoch",    # Evaluierung nach jeder Epoche
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./my-llm")
tokenizer.save_pretrained("./my-llm")

#from transformers import pipeline
#generator = pipeline("text-generation", model="./my-llm")
#print(generator("Einleitung: ", max_length=50))
