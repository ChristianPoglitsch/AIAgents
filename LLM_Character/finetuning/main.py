import logging
import os

# in order to prevent the terminal to be cluttered from all the
# torch/transformers warnings.
import warnings

import torch
from datasets import load_dataset
from generate_data import generate_llm_additional_data
from models import (
    generate_pipe_text,
    generate_text,
    load_base_model,
    load_mistral_instr_model,
    load_pipeline,
)
from peft import PeftModel
from training import train_mistral, train_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from util import get_formatted_prompt, print_generated_text

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# QUESTION: why does the pipeline function perform better than the
# generate_text function. What kind of pre/processing is being missed?


def load_and_train_mistral_example():
    # instruct_tune_dataset1 = generate_additional_data()
    instruct_tune_dataset2 = generate_llm_additional_data()
    model, tokenizer = load_mistral_instr_model()

    prompts = ["Who the hell is ibrahim?"]

    generation_config = GenerationConfig(temperature=0.1)
    max_length = 50

    pipe = load_pipeline(model, tokenizer, max_length)

    for prompt in prompts:
        text1 = generate_pipe_text(pipe, f"<s>[INST] {prompt} [/INST]")
        text2 = generate_text(prompt, model, tokenizer, generation_config, max_length)

        print("\n " + prompt)
        print_generated_text("before fine tuning", prompt, text1, text2)

    # fine tuning
    trainer = train_mistral(model, tokenizer, instruct_tune_dataset2)
    model = trainer.model
    tokenizer = trainer.tokenizer
    # pipe = load_pipeline(model, tokenizer, max_length)

    text1 = generate_pipe_text(pipe, f"<s>[INST] {prompt} [/INST]")
    text2 = generate_text(prompt, model, tokenizer, generation_config, max_length)

    for prompt in prompts:
        text1 = generate_pipe_text(pipe, f"<s>[INST] {prompt} [/INST]")
        text2 = generate_text(prompt, model, tokenizer, generation_config, max_length)

        print_generated_text("\n after fine tuning", prompt, text1, text2)

    del tokenizer
    torch.cuda.empty_cache()


def load_mistral_example():
    model, tokenizer = load_mistral_instr_model()

    adapters_id = "trained\\Mistral-7b-v2-finetune"
    model = PeftModel.from_pretrained(model, adapters_id)

    question = "What is the research focus of Christian Poglitsch from Austria?"
    generation_config = GenerationConfig(temperature=0.1)

    text1 = generate_text(question, model, tokenizer, generation_config, 50)
    print_generated_text("\n after fine tuning", question, text1, "")
    
    del tokenizer
    torch.cuda.empty_cache()


def run_formatting_example(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,  # Lade Modell 4-Bit quantisiert
        torch_dtype=torch.bfloat16,  # Verwende BFloat16-Datentyp für Berechnungen
        device_map="auto",  # Weise Modellgewichtungen automatisch zu: GPU > CPU > Festplatte
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="right", use_fast=False
    )

    torch.manual_seed(1337)

    generation_config = GenerationConfig(
        temperature=0.1,
        # Diversity of the generated text , "greedy vs random" "exploration vs
        # exploitation" choice for probability of the prediction of the next
        # token,  (<1 makes the results more deterministic, while >1 increases
        # randomness in choosing the next token)
        # Select the next token from the 40 most likely next tokens (kNN
        # strategy)
        top_k=40,
        top_p=0.75,  # Choose the next output token from a subset of all likely next tokens where the cumulative probability of the subset is greater than 0.75. Since top_k is also defined, the subset contains at most 40 tokens.
    )

    prompt = "Question: How many federal states does Germany have? Answer:"
    generated_text = generate_text(
        prompt, model, tokenizer, generation_config, 30
    )  # .split("Answer:")[1]
    # loop
    print_generated_text("", prompt, generated_text, "")

    prompt = "I have extreme pain in my lower back."
    formatted_prompt = get_formatted_prompt(prompt)
    generated_text = generate_text(formatted_prompt, model, tokenizer, None, 60)
    print_generated_text("", prompt, generated_text, "")

    # formatted_prompt = get_formatted_prompt(prompt)
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=60)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    del tokenizer
    torch.cuda.empty_cache()


def run_train_model_example(model_id, trained_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="right", use_fast=False
    )

    def preprocess_function(sample):
        MAX_SEQUENCE_LENGTH = 512
        preprocessed_prompt = ""

        # Jeder Turn besteht aus menschlicher Anweisung und Antwort vom
        # Assistenten
        turn_delimiter = "### Human:"
        turns = [
            turn_delimiter + turn
            for turn in sample["text"].split(turn_delimiter)
            if turn
        ]
        for turn in turns:
            # Verarbeite Turn nur wenn Antwort vom Assistenten ebenfalls
            # enthalten ist
            if "### Assistant:" in turn:
                # Jeder Turn beginnt mit einem Satzbeginn- und endet mit einem
                # Satzende-Token
                preprocessed_prompt += tokenizer.bos_token + turn + tokenizer.eos_token
        result = tokenizer(
            preprocessed_prompt,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    # Datensatz vorverarbeiten
    dataset = load_dataset("thisserand/health_care_german", split="train")
    dataset = dataset.map(preprocess_function, remove_columns=["text"])

    # ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Modell wird 4-Bit quantisiert geladen
        bnb_4bit_use_double_quant=True,
        # Quantisiuerungskonstanten werden ebenfalls quantisiert (reduziert
        # Speicherbedarf)
        bnb_4bit_quant_type="nf4",
        # spezifischer Datentyp 'nf4' wird als 4-Bit Datentyp verwendet
        # (optimal für normalverteilte Modellgewichtungen)
        bnb_4bit_compute_dtype=torch.bfloat16,  # Datentyp 'bfloat16' wird für Berechnungen verwendet. Hierfür werden 4-Bit Modellgewichtungen bzw. Teile der Gewichtungen zur Laufzeit zu BFloat16 dequantisiert sodass Matrizenmultiplikation mit 16-bit Genauigkeit durchgeführt werden können
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.torch_dtype = torch.bfloat16

    _ = train_model(model, tokenizer, dataset, trained_path)
    
    del tokenizer
    torch.cuda.empty_cache()



def run_trained_model(model_id, trained_path):
    adapters_id = trained_path
    base_model = load_base_model(model_id)
    model = PeftModel.from_pretrained(base_model, adapters_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="right", use_fast=False
    )

    prompt = "Ich habe extreme Schmerzen im unteren Rücken."
    prompt = "The capital of Austria?"
    formatted_prompt = get_formatted_prompt(prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    #outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=300)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.2,  # 1.0
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
    )
    generation_config.eos_token_id = tokenizer.eos_token_id

    outputs = model.generate(input_ids=inputs["input_ids"], generation_config=generation_config)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# --- main function ---
if __name__ == "__main__":
    path = os.getcwd()
    # Extracting all the contents in the directory corresponding to path
    l_files = os.listdir(path)

    # load_and_train_mistral_example()
    # load_mistral_example()

    model_id = "openlm-research/open_llama_7b_v2"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    trained_path = "trained\\Mistral-7b-v2-finetune"
    trained_path = "trained/health_care_german"
    trained_path = "trained\\Mistral-7b-v3-finetune"

    # run_formatting_example(model_id)
    # run_train_model_example(model_id, trained_path)
    run_trained_model(model_id, trained_path)
