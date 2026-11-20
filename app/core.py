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

