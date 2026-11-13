"""Optional real LoRA SFT; requires pip install -e '.[train]' and adequate hardware.

This is not executed by default or CI. Training a local model requires downloading
weights and may be impractical on CPU-only machines. Never train on evaluation cases.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face causal LM ID available for your hardware")
    parser.add_argument("--train", default="data/train.json")
    parser.add_argument("--output", default="outputs/toolsmith-lora")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer
    from toolsmith.core import format_prompt
    records = json.loads(Path(args.train).read_text())
    if not records:
        raise ValueError("Training dataset is empty")
    if not all("prompt" in x and "expected" in x for x in records):
        raise ValueError("Each record requires prompt and expected")
    strings = [format_prompt(x)+json.dumps(x["expected"],sort_keys=True) for x in records]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    config = LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,task_type="CAUSAL_LM")
    dataset = Dataset.from_dict({"text":strings})
    arguments = TrainingArguments(output_dir=args.output,num_train_epochs=args.epochs,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=2e-4,report_to="none",save_strategy="epoch",logging_steps=1)
    trainer = SFTTrainer(model=model,train_dataset=dataset,peft_config=config,args=arguments,processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output)
    print("Saved adapter:",args.output)

if __name__ == "__main__":
    main()
