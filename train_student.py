import unsloth
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# 1. Load Student Base Model in 4-bit (~2.2 GB baseline VRAM)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

# 2. Attach LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Format Dataset Using Qwen Chat Template
def format_chat(batch):
    formatted_texts = []
    for conv in batch["conversations"]:
        messages = [
            {"role": "user" if msg["from"] == "human" else "assistant", "content": msg["value"]}
            for msg in conv
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
    return {"text": formatted_texts}

raw_data = load_dataset("json", data_files="qwen_distillation_data.jsonl")["train"]
formatted_data = raw_data.map(format_chat, batched=True)

# 4. Training Arguments Configured for RTX 4060 (8 GB)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    output_dir="./distill_checkpoints",
    save_strategy="no",
)

# 5. Initialize Trainer (uses processing_class for newer TRL)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

print("Starting distillation fine-tuning on RTX 4060...")
trainer.train()

# 6. Save LoRA Adapters and Export directly to 8-bit GGUF
print("Exporting model to GGUF format...")
model.save_pretrained("qwen_distilled_3b_lora")
tokenizer.save_pretrained("qwen_distilled_3b_lora")

model.save_pretrained_gguf("qwen_distilled_3b_q8", tokenizer, quantization_method="q8_0")
print("Finished! GGUF file generated.")