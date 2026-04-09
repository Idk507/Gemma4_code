# ==================== INSTALLATION (run once) ====================
# MacOS / Linux / WSL:
# curl -fsSL https://unsloth.ai/install.sh | sh

# Windows PowerShell:
# irm https://unsloth.ai/install.ps1 | iex

# Then in your notebook/script:
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
!pip install -q trl datasets

# ==================== FULL FINE-TUNING SCRIPT ====================
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# 1. Load model (choose any Gemma 4 variant)
model_name = "google/gemma-4-E4B-it"   # Best balance: E4B (or "google/gemma-4-E2B-it", "unsloth/gemma-4-26B-A4B-it", etc.)
max_seq_length = 8192

model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    dtype=None,                    # Auto-detect (bf16 on Ampere+)
    max_seq_length=max_seq_length,
    load_in_4bit=True,             # QLoRA
    full_finetuning=False,         # LoRA by default
)

# 2. Add LoRA adapters (Gemma 4 specific config)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Freeze vision/audio towers for text-only fine-tuning
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,                               # LoRA rank (16 for more capacity)
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# 3. Chat template (Gemma 4 uses special thinking mode template)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

# 4. Load & format dataset (any chat-format dataset works)
dataset = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")  # Example; replace with your dataset

# Optional: standardize if your data is in different formats
# from unsloth.chat_templates import standardize_data_formats
# dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        .removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Trainer (SFT)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,      # Increase if you have more VRAM
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,                       # Change to your needs (or num_train_epochs)
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",                   # or "wandb"
    ),
)

# Only train on assistant responses (standard for chat models)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

# 6. Train!
trainer_stats = trainer.train()

# 7. Save & push (LoRA + merged GGUF)
model.save_pretrained_gguf("gemma-4-E4B-it-finetuned", tokenizer, quantization_method="q4_k_m")
model.push_to_hub_gguf("YOUR_USERNAME/gemma-4-E4B-it-finetuned", tokenizer, quantization_method="q4_k_m")
