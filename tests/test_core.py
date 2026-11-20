"""Tests for the five project milestones. Model-download tests are intentionally avoided in CI."""
import json
import pytest
from app import core


def test_milestone_1_has_four_categories_and_valid_examples():
    groups = core.milestone_1()
    assert set(groups) == set(core.CATEGORIES)
    assert all(len(items) >= 10 for items in groups.values())
    assert all(item["text"].strip() for items in groups.values() for item in items)


def test_milestone_2_declares_one_special_token_per_category():
    assert len(core.SPECIAL_TOKENS) == 4
    assert all(token.startswith("<|") and token.endswith("|>") for token in core.SPECIAL_TOKENS)


def test_milestone_3_dataset_masks_prompt_tokens():
    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [len(word) for word in text.split()]}
    dataset = core.ReportDataset([{"text":"leaves have spots", "label":"PEST_DISEASE"}], FakeTokenizer(), max_length=20)
    item = dataset[0]
    assert (item["labels"] != -100).any()
    assert (item["labels"] == -100).any()


def test_milestone_4_rejects_empty_report():
    with pytest.raises(ValueError):
        core.classify_report("   ")


def test_milestone_5_reads_nonempty_lines_and_writes_json(tmp_path, monkeypatch):
    source = tmp_path / "reports.txt"
    target = tmp_path / "results.json"
    source.write_text("first report\n\nsecond report\n", encoding="utf-8")
    monkeypatch.setattr(core, "classify_report", lambda text: {"report": text, "category": "SOIL_NUTRIENT"})
    result = core.classify_file(source, target)
    assert [x["report"] for x in result] == ["first report", "second report"]
    assert json.loads(target.read_text()) == result
