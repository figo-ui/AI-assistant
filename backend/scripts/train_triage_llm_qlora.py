import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def build_preflight(base_model: str) -> dict:
    import platform
    import psutil

    try:
        import torch
    except Exception:
        torch = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "base_model": base_model,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch else 0,
        "transformers_no_torchvision": os.getenv("TRANSFORMERS_NO_TORCHVISION", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA / Unsloth triage LLM fine-tuning entrypoint.")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base instruct model. Replace with meta-llama/Llama-3.1-8B-Instruct if available.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        default=str(Path(__file__).resolve().parents[1] / "data" / "triage_llm_dataset.jsonl"),
        help="Prepared JSONL dataset from prepare_triage_llm_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "models" / "triage_llm_adapter"),
        help="Adapter output directory.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="By default the script refuses 7B QLoRA training when CUDA is unavailable.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    dataset_path = Path(args.dataset_jsonl)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(args.base_model)
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

    if not args.allow_cpu and not preflight["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available on this machine. Refusing to start local 7B QLoRA training without --allow-cpu."
        )
    if not args.allow_cpu and preflight["ram_gb"] < 24:
        raise RuntimeError(
            f"Only {preflight['ram_gb']} GB RAM detected. Local 7B fine-tuning requires substantially more memory."
        )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    ds = ds.train_test_split(test_size=0.02, seed=int(args.seed))

    try:
        from unsloth import FastLanguageModel  # type: ignore
        from transformers import TrainingArguments
        from trl import SFTTrainer

        unsloth_available = True
    except Exception:
        unsloth_available = False

    if unsloth_available:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=int(args.max_seq_length),
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )

        def format_messages(batch):
            texts = []
            for messages in batch["messages"]:
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            return {"text": texts}

        train_ds = ds["train"].map(format_messages, batched=True)
        eval_ds = ds["test"].map(format_messages, batched=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            max_seq_length=int(args.max_seq_length),
            args=TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=int(args.batch_size),
                per_device_eval_batch_size=int(args.batch_size),
                gradient_accumulation_steps=int(args.grad_accum),
                learning_rate=float(args.learning_rate),
                num_train_epochs=int(args.epochs),
                logging_steps=20,
                evaluation_strategy="steps",
                eval_steps=200,
                save_steps=200,
                save_total_limit=2,
                bf16=True,
                report_to="none",
            ),
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mode = "unsloth"
    else:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        def format_messages(batch):
            texts = []
            for messages in batch["messages"]:
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            return {"text": texts}

        train_ds = ds["train"].map(format_messages, batched=True, remove_columns=ds["train"].column_names)
        eval_ds = ds["test"].map(format_messages, batched=True, remove_columns=ds["test"].column_names)

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=int(args.max_seq_length))

        train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            args=TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=int(args.batch_size),
                per_device_eval_batch_size=int(args.batch_size),
                gradient_accumulation_steps=int(args.grad_accum),
                learning_rate=float(args.learning_rate),
                num_train_epochs=int(args.epochs),
                logging_steps=20,
                evaluation_strategy="steps",
                eval_steps=200,
                save_steps=200,
                save_total_limit=2,
                fp16=True,
                report_to="none",
            ),
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mode = "transformers"

    summary = {
        "base_model": args.base_model,
        "dataset_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "epochs": int(args.epochs),
        "mode": mode,
        "train_rows": int(len(ds["train"])),
        "eval_rows": int(len(ds["test"])),
        "preflight": preflight,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
