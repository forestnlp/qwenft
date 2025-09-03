import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from transformers import Trainer

def main():
    # Model and tokenizer names
    model_name = "Qwen/Qwen3-8B-Instruct"

    # 1. Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Abirate/english_quotes", split="train")

    # 2. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Configure model for 4-bit quantization
    print("Configuring model for 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 4. Load model with quantization config
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # Automatically maps layers to available devices (GPU/CPU)
        trust_remote_code=True
    )
    model.config.use_cache = False # Recommended for training

    # 5. Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8, # Rank of the update matrices. Lower rank means fewer parameters to train.
        lora_alpha=16, # Alpha parameter for scaling.
        lora_dropout=0.05, # Dropout probability for LoRA layers.
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Modules to apply LoRA to.
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Set training arguments
    print("Setting training arguments...")
    training_arguments = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=-1, # If set, overrides num_train_epochs
        fp16=True, # Use 16-bit precision for training
    )

    # 7. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data])}
    )

    # 8. Start training
    print("Starting training...")
    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    main()
