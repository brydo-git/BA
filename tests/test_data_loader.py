import io
import pandas as pd
import pytest
from unittest.mock import patch
from data_loader import DataLoader


class TestMakeUnique:
    def test_no_duplicates_unchanged(self):
        assert DataLoader._make_unique(["a", "b", "c"]) == ["a", "b", "c"]

    def test_single_duplicate_gets_suffix(self):
        assert DataLoader._make_unique(["a", "a"]) == ["a", "a_1"]

    def test_triple_duplicate(self):
        assert DataLoader._make_unique(["a", "a", "a"]) == ["a", "a_1", "a_2"]

    def test_non_string_columns_converted(self):
        assert DataLoader._make_unique([1, 1, 2]) == ["1", "1_1", "2"]

    def test_different_names_unaffected(self):
        assert DataLoader._make_unique(["x", "y", "x", "z"]) == ["x", "y", "x_1", "z"]


class TestApplyHeader:
    def test_sets_column_names_from_row(self):
        df = pd.DataFrame([["col_a", "col_b"], [1, 2], [3, 4]])
        result = DataLoader.apply_header(df, row=0)
        assert list(result.columns) == ["col_a", "col_b"]

    def test_drops_header_row(self):
        df = pd.DataFrame([["col_a", "col_b"], [10, 20], [30, 40]])
        result = DataLoader.apply_header(df, row=0)
        assert len(result) == 2
        assert result.iloc[0, 0] == 10

    def test_non_zero_header_row(self):
        # rows above the header and the header row itself are dropped
        df = pd.DataFrame([["meta", "info"], ["col_a", "col_b"], [1, 2]])
        result = DataLoader.apply_header(df, row=1)
        assert list(result.columns) == ["col_a", "col_b"]
        assert len(result) == 1  # only row 2 (1/2) remains

    def test_deduplicates_columns(self):
        df = pd.DataFrame([["x", "x", "y"], [1, 2, 3]])
        result = DataLoader.apply_header(df, row=0)
        assert "x" in result.columns
        assert "x_1" in result.columns

    def test_resets_index(self):
        df = pd.DataFrame([["a", "b"], [1, 2], [3, 4]])
        result = DataLoader.apply_header(df, row=0)
        assert list(result.index) == [0, 1]


class TestLoad:
    def test_raises_for_xls(self):
        with pytest.raises(ValueError, match="nicht unterstützt"):
            DataLoader.load(b"data", "file.xls")

    def test_raises_for_unknown_extension(self):
        with pytest.raises(ValueError):
            DataLoader.load(b"data", "file.json")

    def test_delegates_to_cache_for_csv(self):
        fake_df = pd.DataFrame({"a": [1]})
        with patch("data_loader._cached_load", return_value=fake_df) as mock:
            result = DataLoader.load(b"a\n1", "test.csv")
            mock.assert_called_once_with(b"a\n1", "test.csv", 0)
            assert result is fake_df

    def test_delegates_to_cache_for_xlsx(self):
        fake_df = pd.DataFrame({"x": [1]})
        with patch("data_loader._cached_load", return_value=fake_df) as mock:
            result = DataLoader.load(b"fakebytes", "test.xlsx")
            mock.assert_called_once_with(b"fakebytes", "test.xlsx", 0)
            assert result is fake_df

    def test_delegates_to_cache_for_xlsb(self):
        fake_df = pd.DataFrame({"x": [1]})
        with patch("data_loader._cached_load", return_value=fake_df) as mock:
            result = DataLoader.load(b"fakebytes", "test.xlsb")
            mock.assert_called_once_with(b"fakebytes", "test.xlsb", 0)
            assert result is fake_df

    def test_passes_through_sheet_name(self):
        fake_df = pd.DataFrame({"x": [1]})
        with patch("data_loader._cached_load", return_value=fake_df) as mock:
            DataLoader.load(b"fakebytes", "test.xlsx", "Sheet2")
            mock.assert_called_once_with(b"fakebytes", "test.xlsx", "Sheet2")


class TestSheetNames:
    def test_empty_for_csv(self):
        assert DataLoader.sheet_names(b"a\n1", "test.csv") == []

    def test_empty_for_unsupported_extension(self):
        assert DataLoader.sheet_names(b"data", "test.json") == []

    def test_delegates_to_cache_for_xlsx(self):
        with patch("data_loader._cached_sheet_names", return_value=["Sheet1", "Sheet2"]) as mock:
            result = DataLoader.sheet_names(b"fakebytes", "test.xlsx")
            mock.assert_called_once_with(b"fakebytes", "test.xlsx")
            assert result == ["Sheet1", "Sheet2"]

    def test_delegates_to_cache_for_xlsb(self):
        with patch("data_loader._cached_sheet_names", return_value=["Sheet1"]) as mock:
            result = DataLoader.sheet_names(b"fakebytes", "test.xlsb")
            mock.assert_called_once_with(b"fakebytes", "test.xlsb")
            assert result == ["Sheet1"]
