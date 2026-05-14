from pathlib import Path

from domain.feature_dictionary import FEATURE_COLUMNS, GROUP_COLUMN, TARGET_BINARY
from infrastructure.data import load_primary_dataset, load_secondary_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_primary_dataset_schema():
    df = load_primary_dataset(ROOT / "data/raw/heart_disease_uci.csv")
    assert GROUP_COLUMN in df.columns
    assert TARGET_BINARY in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert set(df[TARGET_BINARY].dropna().unique()) <= {0, 1}
    assert df[GROUP_COLUMN].nunique() >= 2


def test_secondary_dataset_deduplication():
    original = load_secondary_dataset(ROOT / "data/raw/heart.csv", deduplicate=False)
    deduped = load_secondary_dataset(ROOT / "data/raw/heart.csv", deduplicate=True)
    assert len(original) > len(deduped)
    assert len(deduped) == 302

