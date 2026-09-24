import unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_distilled_3b_lora",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

print("Merging LoRA weights and exporting directly to Q8_0 GGUF...")
model.save_pretrained_gguf(
    "qwen_distilled_3b_q8",
    tokenizer,
    quantization_method="q8_0"
)
print("Finished! GGUF created successfully.")
