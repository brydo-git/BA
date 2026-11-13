import pandas as pd
import pytest
from classifier import ColumnClassifier


class TestIsNumeric:
    def test_integer_series(self):
        assert ColumnClassifier.is_numeric(pd.Series([1, 2, 3]))

    def test_float_series(self):
        assert ColumnClassifier.is_numeric(pd.Series([1.1, 2.2]))

    def test_string_series(self):
        assert not ColumnClassifier.is_numeric(pd.Series(["a", "b", "c"]))

    def test_numeric_strings_are_numeric(self):
        assert ColumnClassifier.is_numeric(pd.Series(["1", "2", "3"]))

    def test_mixed_with_some_numbers(self):
        assert ColumnClassifier.is_numeric(pd.Series(["1", "abc", "3"]))

    def test_all_nan(self):
        assert not ColumnClassifier.is_numeric(pd.Series([None, None]))


class TestIsCategorical:
    def test_low_cardinality_is_categorical(self):
        s = pd.Series(["A", "B", "A", "B", "C"])
        assert ColumnClassifier.is_categorical(s) is True

    def test_high_cardinality_is_not_categorical(self):
        s = pd.Series([str(i) for i in range(1000)])
        assert ColumnClassifier.is_categorical(s) is False

    def test_exactly_at_max_abs_boundary(self):
        s = pd.Series([str(i) for i in range(100)])
        assert ColumnClassifier.is_categorical(s) is True

    def test_empty_series(self):
        s = pd.Series([], dtype=str)
        assert ColumnClassifier.is_categorical(s) is True


class TestNunique:
    def test_counts_unique(self):
        assert ColumnClassifier.nunique(pd.Series(["A", "B", "A", "C"])) == 3

    def test_ignores_nan(self):
        assert ColumnClassifier.nunique(pd.Series(["A", "B", None])) == 2

    def test_single_value(self):
        assert ColumnClassifier.nunique(pd.Series(["X", "X", "X"])) == 1


class TestNumericCols:
    def test_returns_only_numeric(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        assert ColumnClassifier.numeric_cols(df) == ["a", "c"]

    def test_empty_df(self):
        assert ColumnClassifier.numeric_cols(pd.DataFrame()) == []


class TestCategoricalCols:
    def test_returns_string_columns(self):
        # 101 unique numeric values exceeds max_abs=100 and max_ratio threshold
        df = pd.DataFrame({"cat": ["A"] * 101, "num": list(range(101))})
        result = ColumnClassifier.categorical_cols(df)
        assert "cat" in result
        assert "num" not in result

    def test_numeric_low_cardinality_also_categorical(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1]})
        assert "flag" in ColumnClassifier.categorical_cols(df)


class TestDateCols:
    def test_detects_iso_dates(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "val": [1, 2, 3],
        })
        assert "date" in ColumnClassifier.date_cols(df)
        assert "val" not in ColumnClassifier.date_cols(df)

    def test_ignores_plain_text(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        assert ColumnClassifier.date_cols(df) == []

    def test_empty_df(self):
        assert ColumnClassifier.date_cols(pd.DataFrame()) == []

    def test_detects_calendar_week_column(self):
        df = pd.DataFrame({
            "kw": [202601, 202602, 202653],
            "val": [1, 2, 3],
        })
        assert "kw" in ColumnClassifier.date_cols(df)
        assert "val" not in ColumnClassifier.date_cols(df)

    def test_plain_numeric_column_not_treated_as_date(self):
        # small integers don't match the strict 6-digit YYYYWW pattern
        df = pd.DataFrame({"qty": [1, 2, 3, 4, 5]})
        assert ColumnClassifier.date_cols(df) == []


class TestParseDates:
    def test_parses_normal_dates(self):
        s = pd.Series(["2024-01-01", "2024-01-02"])
        parsed = ColumnClassifier.parse_dates(s)
        assert parsed.notna().all()

    def test_parses_calendar_week_to_monday(self):
        s = pd.Series([202601, 202602, 202653])
        parsed = ColumnClassifier.parse_dates(s)
        assert parsed.notna().all()
        assert all(d.weekday() == 0 for d in parsed)  # Monday
        assert parsed.is_monotonic_increasing

    def test_calendar_week_as_string(self):
        s = pd.Series(["202601", "202602"])
        parsed = ColumnClassifier.parse_dates(s)
        assert parsed.notna().all()

    def test_invalid_week_number_becomes_nat(self):
        # week 99 doesn't exist, but the other 3 values still meet the 70% calendar-week threshold
        s = pd.Series([202699, 202601, 202602, 202603])
        parsed = ColumnClassifier.parse_dates(s)
        assert parsed.isna().sum() == 1
