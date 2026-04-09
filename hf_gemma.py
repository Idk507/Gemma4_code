# !pip install -q -U transformers accelerate datasets trl peft bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from huggingface_hub import login

login()  # or set HF_TOKEN env var

MODEL_ID = "google/gemma-4-E4B-it"

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset (same formatting as above)
dataset = load_dataset("mlabonne/FineTome-100k", split="train[:2000]")

def formatting_func(example):
    # Use apply_chat_template for Gemma 4
    return tokenizer.apply_chat_template(example["conversations"], tokenize=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        output_dir="gemma4-finetune",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    formatting_func=formatting_func,
)

trainer.train()

# Merge & save
model = model.merge_and_unload()
model.save_pretrained("gemma-4-finetuned-merged")
tokenizer.save_pretrained("gemma-4-finetuned-merged")
