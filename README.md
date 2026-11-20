# Agricultural Report Classifier

A command-line project that fine-tunes **GPT-2** to classify short farmer reports into four agricultural categories. It demonstrates special-token extension, causal-language-model fine-tuning with Hugging Face `Trainer`, single-report inference, and batch file processing.

## Categories

| Special category token | Meaning |
|---|---|
| `PEST_DISEASE` | Pest activity, disease symptoms and visible crop damage |
| `WATER_IRRIGATION` | Irrigation failures, water shortage, flooding and drainage |
| `SOIL_NUTRIENT` | Soil quality, pH, nutrient deficiency and soil structure |
| `WEATHER_CROP_STRESS` | Heat, frost, wind, hail, drought and weather-related damage |

Training examples are in `data/training_data.json`; `data/sample_reports.txt` is a small batch input example. These are illustrative synthetic examples, not a field-validated agricultural dataset. A model trained on this small set is a learning demonstration and should not be used as an authoritative farm diagnosis system.

## Setup (Windows PowerShell)

Python 3.10+ is recommended. GPT-2 and PyTorch are sizeable downloads; the first training run downloads the base model from Hugging Face.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Train the model

```powershell
python -m app.cli train
```

For a quick experiment with fewer passes:

```powershell
python -m app.cli train --epochs 2
```

The fine-tuned model and tokenizer are saved under `models/agri-gpt2/`. The `models/` directory is ignored by Git because model weights are large; rerun training after cloning.

## Classify one report

```powershell
python -m app.cli classify --text "Brown spots are spreading across tomato leaves."
```

The command prints JSON containing the report, predicted category, category description and a confidence-like normalized score over the four category-token logits. This score is not calibrated probability or a measure of agronomic certainty.

## Classify a file

Put one report per line in a UTF-8 text file, then run:

```powershell
python -m app.cli batch --input data/sample_reports.txt --output results.json
```

The CLI prints all predictions and writes them to `results.json`.

## Tests

```powershell
pytest -q
```

Tests cover category definitions, special token setup, prompt/target masking, input validation and batch file processing. They do not download GPT-2 in CI.

## How the model works

1. Each category has a dedicated special token, such as `<|PEST_DISEASE|>`.
2. `prepare_model()` adds these tokens to the GPT-2 tokenizer and calls `resize_token_embeddings()` on the model.
3. Each training sequence contains a report prompt followed by its category token. The prompt portion is masked with `-100`, so the training loss focuses on generating the category token.
4. Hugging Face `Trainer` fine-tunes GPT-2 and saves the model/tokenizer.
5. At inference, the model scores the four category tokens as the next token and selects the highest-scoring one.

## Limitations

- The bundled examples are synthetic and small. More diverse, expert-labelled field reports are needed for meaningful performance.
- Reports outside these four categories will still be assigned the closest of the four labels.
- GPT-2 is a generative language model, not a specialist agronomy model. Verify important observations with a qualified agricultural expert.
- CPU training may take several minutes or longer. A GPU can speed it up when configured with a compatible PyTorch build.

## Git milestones (six commits)

1. `Add agricultural categories and sample training data`
2. `Extend GPT-2 tokenizer with agricultural category tokens`
3. `Fine-tune GPT-2 with Hugging Face Trainer`
4. `Add single-report classification CLI`
5. `Add batch report classification from text files`
6. `Add tests and project documentation`

Commit after each milestone as you implement it; the exact push commands are included in the project handoff.
