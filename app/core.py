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

