"""Dataset definitions and GPT-2 fine-tuning / inference pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


BASE_MODEL = "gpt2"
MODEL_DIR = Path("models/agri-gpt2")
DATASET_PATH = Path("data/training_data.json")
CATEGORIES = {
    "PEST_DISEASE": "Pest or disease symptoms and visible crop damage",
    "WATER_IRRIGATION": "Water shortage, excess water, drainage or irrigation problems",
    "SOIL_NUTRIENT": "Soil quality, pH, nutrient deficiency or soil structure observations",
    "WEATHER_CROP_STRESS": "Weather-related crop stress or physical weather damage",
}
SPECIAL_TOKENS = [f"<|{name}|>" for name in CATEGORIES]
PROMPT = "Classify this agricultural field report into one category.\nReport: {report}\nCategory:"


def milestone_1() -> dict[str, list[dict[str, str]]]:
    """Return the labelled examples used for supervised fine-tuning."""
    examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not examples or any(x.get("label") not in CATEGORIES or not x.get("text", "").strip() for x in examples):
        raise ValueError("Training data must contain non-empty text and a known category label.")
    return {name: [x for x in examples if x["label"] == name] for name in CATEGORIES}


def prepare_model(model_name: str = BASE_MODEL):
    """Load GPT-2, add one special token per category, and resize embeddings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def milestone_2(model_name: str = BASE_MODEL):
    return prepare_model(model_name)


class ReportDataset:
    """Small torch Dataset; prompt tokens are masked so loss trains the category token."""
    def __init__(self, examples: list[dict[str, str]], tokenizer, max_length: int = 128):
        import torch
        self.items = []
        for item in examples:
            prompt = PROMPT.format(report=item["text"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(f"<|{item['label']}|>{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            attention = [1] * len(input_ids)
            pad = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad
            labels += [-100] * pad
            attention += [0] * pad
            self.items.append({"input_ids": torch.tensor(input_ids),
                               "attention_mask": torch.tensor(attention),
                               "labels": torch.tensor(labels)})
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]


def train_model(output_dir: str | Path = MODEL_DIR, epochs: int = 8,
                model_name: str = BASE_MODEL, examples: list[dict[str, str]] | None = None):
    """Fine-tune GPT-2 to generate the special token corresponding to each report."""
    examples = examples if examples is not None else [x for group in milestone_1().values() for x in group]
    tokenizer, model = prepare_model(model_name)
    dataset = ReportDataset(examples, tokenizer)
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs, per_device_train_batch_size=4,
        gradient_accumulation_steps=1, learning_rate=5e-5, weight_decay=0.01,
        logging_steps=5, save_strategy="no", report_to="none",
        fp16=False, use_cpu=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    # DataCollatorForLanguageModeling recreates labels; use the labels prepared above instead.
    class PreserveLabels:
        def __call__(self, features):
            import torch
            return {key: torch.stack([f[key] for f in features]) for key in features[0]}
    trainer.data_collator = PreserveLabels()
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return Path(output_dir)


def milestone_3():
    return train_model()


def classify_report(report: str, model_dir: str | Path = MODEL_DIR) -> dict[str, str | float]:
    """Generate a category token from the fine-tuned model for one report."""
    if not report or not report.strip():
        raise ValueError("Report text cannot be empty.")
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError("Fine-tuned model not found. Run: python -m app.cli train")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    prompt = PROMPT.format(report=report.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        next_logits = output.logits[0, -1, :]
        token_ids = [tokenizer.convert_tokens_to_ids(f"<|{name}|>") for name in CATEGORIES]
        category_logits = next_logits[token_ids]
        probs = torch.softmax(category_logits, dim=0)
        best = int(torch.argmax(probs).item())
    label = list(CATEGORIES)[best]
    return {"report": report.strip(), "category": label,
            "description": CATEGORIES[label], "confidence": float(probs[best].item())}


def milestone_4(report: str):
    return classify_report(report)


def classify_file(file_path: str | Path, output_path: str | Path | None = None) -> list[dict]:
    """Classify non-empty lines in a UTF-8 text file, preserving their order."""
    source = Path(file_path)
    reports = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [classify_report(report) for report in reports]
    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def milestone_5(file_path: str | Path):
    return classify_file(file_path)
