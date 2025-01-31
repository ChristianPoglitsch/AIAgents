import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from trl import SFTTrainer


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
    trainer.save_model("trained\\Mistral-7b-v2-finetune")
    # model.eval()
    return trainer


# https://www.e2enetworks.com/blog/a-step-by-step-guide-to-fine-tuning-the-mistral-7b-llm
# issue : PeftSavingCallback could not be imported.


def train_model(model, tokenizer, dataset, trained_path):
    """is not trained with SFFT

    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        dataset (_type_): _description_
        trained_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Prepare model for QLoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    # Configure Low-Rank Adapter (LoRA)
    lora_config = LoraConfig(
        r=64,  # Rank of matrix factorizations
        lora_alpha=16,  # LoRA scaling factor
        target_modules=["_proj"],  # Layers in which adapters are added
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.
    # Parameter-efficient fine-tuning, PEFT,
    # adjusts a small number of key parameters while preserving
    # most of the pretrained model's structure to improve LLM performance.

    # Add adapter to the model
    model = get_peft_model(model, lora_config)

    # Convert added adapters to data type BFloat16
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            # Normalization requires fewer parameters; higher accuracy in
            # normalization ensures more stable fine-tuning
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # ---
    training_arguments = TrainingArguments(
        per_device_train_batch_size=1,  # Batchgröße je GPU
        gradient_accumulation_steps=16,
        # Gradienten werden über zwei Batches (2*16=32) akkumuliert und
        # Low-Rank-Adapter werden erst dann aktualisiert (nicht per Batch);
        # Methode um Speicher-effizient die Batchgröße zu vergößern
        max_steps=1875,
        # maximale Anzahl an Trainingsiterationen (pro Iteration werden
        # Modellgewichtungen optimiert) -- 1875
        learning_rate=2e-4,
        # Lernrate (2e-4 liefert gute Ergebnisse für 7B-LLMs)
        warmup_ratio=0.03,
        # Anzahl Iterationen in denen tatsächliche Lernrate linear von 0 zu
        # definierter Lernrate erhöht wird (stabilisiert Trainingsprozess)
        lr_scheduler_type="constant",
        # Nachdem Warmup-Phase abgeschlossen ist bleibt die Lernrate konstant
        bf16=True,
        # sorgt für 16-Bit (anstatt 32-Bit) Fine-Tuning, betrifft zu
        # optimierende Low-Rank Adapter sowie die zur Optimierung benötigten
        # Gradienten
        logging_steps=10,  # Interval in dem Trainingsfortschritt geloggt wird
        output_dir="outputs",  # Ausgabeordner in dem die Low Rank Adapter gespeichert werden
        optim="paged_adamw_32bit",
        # Zu verwendender Optimizer (aktualisiert/optimiert Modellgewichtungen
        # während des Fine-Tunings) -- paged_adamw_32bit
        gradient_checkpointing=True,
        # Verringert Speicherbedarf und vermeidet OOM-Errors, macht das
        # Training jedoch langsamer
        max_grad_norm=0.3,
        # Vermeidet das Gradienten während des Trainings zu groß werden (dient
        # der Stabilisierung des Trainings)
    )

    training_arguments.max_steps = 200
    training_arguments.output_dir = trained_path
    training_arguments.optim = "paged_adamw_8bit"

    # code updates 22.3.2024
    tokenizer.pad_token = "0"

    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
        # FIXME: callbacks=[PeftSavingCallback]
    )

    # Disabling the cache reduces memory usage. However, it should be
    # reactivated during inference as it speeds up text generation
    model.config.use_cache = False

    trainer.train()

    # Saving the fine-tuned Low-Rank Adapters
    trainer.save_model()
    return trainer.model


if __name__ == "__main__":
    # fine tuning

    from datasets import load_dataset
    from models import load_mistral_instr_model

    model, tokenizer = load_mistral_instr_model()

    instruct_tune_dataset = load_dataset("mwitiderrick/lamini_mistral", split="train")
    #instruct_tune_dataset = generate_additional_data()
    train_mistral(model, tokenizer, instruct_tune_dataset)

    # FIXME:
    # train_model(model, tokenizer, instruct_tune_dataset, "./output")
