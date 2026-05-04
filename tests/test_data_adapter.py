"""
Tests for src/data_adapter.py.

Verifies that:
1. The adapter output exactly matches the schema expected by data_loader.
2. Status filtering removes sold/pending rows.
3. Critical-field cleaning removes rows with missing price/sqft/zip/etc.
4. Synthesized fields (property_id, year_built, condition, rent) are correct.
5. days_on_market falls back to 30 when prev_sold_date is missing.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.data_adapter import (
    load_kaggle_realtor,
    TARGET_COLUMNS,
    RENT_TO_PRICE,
    DEFAULT_YEAR_BUILT,
    CONDITION_TIERS,
)
from src.data_loader import REQUIRED_COLUMNS


def _write_raw(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a raw Kaggle-format CSV fixture and return its path."""
    raw_cols = [
        "brokered_by", "status", "price", "bed", "bath", "acre_lot",
        "street", "city", "state", "zip_code", "house_size", "prev_sold_date",
    ]
    df = pd.DataFrame(rows, columns=raw_cols)
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return p


def _good_row(**overrides):
    """Return a baseline 'clean' row that should pass all filters."""
    base = {
        "brokered_by": 1234,
        "status": "for_sale",
        "price": 250_000,
        "bed": 3,
        "bath": 2.0,
        "acre_lot": 0.15,
        "street": "100 Oak St",
        "city": "Phoenix",
        "state": "AZ",
        "zip_code": 85001,
        "house_size": 1500,
        "prev_sold_date": "2022-01-15",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------

class TestSchema:
    def test_output_has_exact_required_columns(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row() for _ in range(5)])
        df = load_kaggle_realtor(path)
        assert list(df.columns) == TARGET_COLUMNS

    def test_output_matches_data_loader_schema(self, tmp_path):
        """The adapter output must satisfy data_loader's REQUIRED_COLUMNS."""
        path = _write_raw(tmp_path, [_good_row() for _ in range(5)])
        df = load_kaggle_realtor(path)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"missing required column: {col}"


# ---------------------------------------------------------------------------
# Filtering rules
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_drops_sold_status(self, tmp_path):
        rows = [
            _good_row(status="for_sale"),
            _good_row(status="sold"),
            _good_row(status="ready_to_build"),
        ]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        assert len(df) == 1

    def test_drops_missing_price(self, tmp_path):
        rows = [_good_row(), _good_row(price=None)]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        assert len(df) == 1

    def test_drops_missing_sqft(self, tmp_path):
        rows = [_good_row(), _good_row(house_size=None)]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        assert len(df) == 1

    def test_drops_missing_zip(self, tmp_path):
        rows = [_good_row(), _good_row(zip_code=None)]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        assert len(df) == 1

    def test_drops_outlier_price(self, tmp_path):
        rows = [_good_row(), _good_row(price=10_000_000)]  # too high
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Synthesized fields
# ---------------------------------------------------------------------------

class TestSynthesizedFields:
    def test_property_ids_are_stable_format(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row() for _ in range(3)])
        df = load_kaggle_realtor(path)
        assert df["property_id"].tolist() == ["K000000", "K000001", "K000002"]

    def test_year_built_defaults(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row()])
        df = load_kaggle_realtor(path)
        assert df["year_built"].iloc[0] == DEFAULT_YEAR_BUILT

    def test_rent_is_seven_tenths_of_a_percent_of_price(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row(price=400_000)])
        df = load_kaggle_realtor(path)
        expected = round(400_000 * RENT_TO_PRICE, 0)
        assert df["estimated_monthly_rent"].iloc[0] == expected

    def test_condition_in_allowed_values(self, tmp_path):
        rows = [_good_row(price=p) for p in (80_000, 200_000, 500_000)]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        for cond in df["property_condition"]:
            assert cond in CONDITION_TIERS

    def test_dom_defaults_when_prev_sold_missing(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row(prev_sold_date=None)])
        df = load_kaggle_realtor(path)
        assert df["days_on_market"].iloc[0] == 30


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

class TestGeocoding:
    def test_lat_lon_populated_for_known_zip(self, tmp_path):
        path = _write_raw(tmp_path, [_good_row(zip_code=85001)])
        df = load_kaggle_realtor(path)
        # Phoenix is around 33.4 N, -112.1 W.
        assert 33.0 < df["latitude"].iloc[0] < 34.0
        assert -113.0 < df["longitude"].iloc[0] < -111.0

    def test_unknown_zip_drops_row(self, tmp_path):
        # ZIPs beginning with 00000 are unallocated -> pgeocode returns NaN.
        rows = [_good_row(), _good_row(zip_code=99999)]
        path = _write_raw(tmp_path, rows)
        df = load_kaggle_realtor(path)
        # The known 85001 row should survive; the bogus one should drop.
        assert (df["zip"] == 85001).all()
