import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

file_path = "example.txt"  # Change this to your file's path

# Open and read the file
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

model_id = "deepseek-ai/deepseek-llm-7b-chat"
model_id = "trained/deepseek-llm-7b-chat_merged"
model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"

#model_id = "trained/Mistral-7B-Instruct-v0.3_merged"
#model_id = "trained/deepseek-llm-7b-chat_merged"
#model_id = "trained\\Teuken-7B-instruct-research-v0.4_merged"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
#model.generation_config = GenerationConfig.from_pretrained(model_id)
#model.generation_config.pad_token_id = model.generation_config.eos_token_id

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Ensures stable quantization
    bnb_4bit_use_double_quant=True,  # Improves efficiency
)

# Load Base Model with Auto Device Placement
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,  
    device_map="auto"  # Automatically assigns to GPU/CPU
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "user", "content": content}
]
#input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

device = "cuda"
input_tensor = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(device)  # tokenize=False)

generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.2,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=100,
)
generation_config.eos_token_id = tokenizer.eos_token_id

outputs = model.generate(input_tensor, generation_config=generation_config)

#outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
