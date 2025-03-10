import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from trl import SFTTrainer

from datasets import load_dataset
from datasets import load_from_disk

def train_mistral(model, tokenizer, instruct_tune_dataset) -> SFTTrainer:
    """is trained with SFFT

    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        instruct_tune_dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        per_device_train_batch_size=4,  # Batch size per GPU
        # gradient_accumulation_steps=16, # Gradients are accumulated over two
        # batches (2*16=32) and Low-Rank Adapters are updated only then (not
        # per batch); method to increase batch size in a memory-efficient
        # manner.
        # num_train_epochs=5,

        max_steps=500,  # comment out this line if you want to train in epochs - 100+ recommended
        save_strategy="epoch",
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=1010,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2e-4,
        warmup_steps=0,
        # warmup_ratio =0, # Number of iterations in which the actual learning
        # rate is linearly increased from 0 to the defined learning rate
        # (stabilizes the training process).
        lr_scheduler_type="constant",
        bf16=True,
        # Ensures 16-bit (instead of 32-bit) fine-tuning, affecting both the
        # Low-Rank Adapters to be optimized and the gradients required for
        # optimization.
        logging_steps=10,  # Interval at which training progress is logged
        output_dir="./output",
        # optim="paged_adamw_32bit", # Optimizer to be used (updates/optimizes model weights during fine-tuning) -- paged_adamw_32bit
        # gradient_checkpointing=True, # Verringert Speicherbedarf und vermeidet OOM-Errors, macht das Training jedoch langsamer
        # max_grad_norm = 0.3
    )

    # max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        train_dataset=instruct_tune_dataset,
        peft_config=peft_config,
        max_seq_length=None,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=args,
        packing=False,
    )

    trainer.train()
    trainer.save_model("trained\\Mistral-7b-v3-finetune")
    # model.eval()
    return trainer

def format_prompts(examples):
    formatted_examples = {
        "prompt": [],
        "response": []
    }
    
    for i in range(len(examples["input"])):
        prompt = examples["input"][i]
        response = examples["output"][i]
        
        # Format the prompt for Mistral-style training
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        formatted_examples["prompt"].append(formatted_prompt)
        formatted_examples["response"].append(response)

    #return formatted_examples
    return {"text": [f"{p} {r}" for p, r in zip(formatted_examples["prompt"], formatted_examples["response"])]}

if __name__ == "__main__":
    # fine tuning

    from datasets import load_dataset
    from models import load_mistral_instr_model    

    file_name = 'training.csv_'
    dataset = load_from_disk(file_name)

    dataset = dataset.map(format_prompts, batched=True)

    model, tokenizer = load_mistral_instr_model()
    train_mistral(model, tokenizer, dataset)
