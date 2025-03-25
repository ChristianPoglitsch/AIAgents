import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

device_map = {"": 0}


model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_id = "deepseek-ai/deepseek-llm-7b-chat"

adapters_id = "trained\\Mistral-7b-v3-finetune"
adapters_id = "trained\\deepseek-llm-7b-chat"

model_id_merged = "trained/Mistral-7B-Instruct-v0.3_merged"
model_id_merged = "trained/deepseek-llm-7b-chat_merged"

#quantization_config = BitsAndBytesConfig(load_in_4bit=True)
#
#model = AutoModelForCausalLM.from_pretrained(  # device_map="auto"
#    model_id,
#    quantization_config=quantization_config,
#    torch_dtype=torch.bfloat16, # token=' '
#)
#
##model = PeftModel.from_pretrained(model, adapters_id)
#model.load_adapter(adapters_id)
#
#model= model.merge_and_unload()

model = AutoPeftModelForCausalLM.from_pretrained(adapters_id, device_map=device_map, torch_dtype=torch.bfloat16)

merged_model = model.merge_and_unload()
# save merged model
merged_model.save_pretrained(model_id_merged)
