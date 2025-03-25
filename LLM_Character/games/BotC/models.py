import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)


def load_pipeline(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int
):
    return pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_length
    )


def generate_pipe_text(pipe, prompt, max_length=100000, temperature=0.7):
    result = pipe(prompt, max_length=max_length, temperature=temperature)
    return result[0]["generated_text"]


def generate_text(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    max_num_token: int,
):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    output_tokens = model.generate(
        input_ids=inputs.to(model.device),
        generation_config=generation_config,
        max_new_tokens=max_num_token,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    output_text = tokenizer.decode(output_tokens.squeeze(), skip_special_tokens=True)
    return output_text


def load_base_model(model_id: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )


def load_mistral_instr_model():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # "mistralai/Mistral-7B-Instruct-v0.1
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=nf4_config, use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def load_model(model_id: str):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=nf4_config, use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
