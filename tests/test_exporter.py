import io
import pandas as pd
import pytest
from exporter import Exporter


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


class TestToCsv:
    def test_returns_bytes(self, df):
        assert isinstance(Exporter.to_csv(df), bytes)

    def test_roundtrip_with_index(self, df):
        data = Exporter.to_csv(df, index=True)
        result = pd.read_csv(io.BytesIO(data), index_col=0)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_roundtrip_without_index(self, df):
        data = Exporter.to_csv(df, index=False)
        result = pd.read_csv(io.BytesIO(data))
        assert list(result.columns) == ["a", "b"]

    def test_utf8_umlaut_encoding(self):
        df = pd.DataFrame({"ä": [1], "ö": [2], "ü": [3]})
        text = Exporter.to_csv(df).decode("utf-8")
        assert "ä" in text and "ö" in text and "ü" in text

    def test_values_preserved(self, df):
        data = Exporter.to_csv(df, index=False)
        result = pd.read_csv(io.BytesIO(data))
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == [4.0, 5.0, 6.0]


class TestToExcel:
    def test_returns_bytes(self, df):
        result = Exporter.to_excel(df)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_roundtrip_with_index(self, df):
        data = Exporter.to_excel(df, sheet="Data")
        result = pd.read_excel(io.BytesIO(data), sheet_name="Data", index_col=0)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_roundtrip_without_index(self, df):
        data = Exporter.to_excel(df, index=False)
        result = pd.read_excel(io.BytesIO(data))
        assert list(result.columns) == ["a", "b"]

    def test_custom_sheet_name(self, df):
        data = Exporter.to_excel(df, sheet="MySheet")
        xl = pd.ExcelFile(io.BytesIO(data))
        assert "MySheet" in xl.sheet_names

    def test_values_preserved(self, df):
        data = Exporter.to_excel(df, index=False)
        result = pd.read_excel(io.BytesIO(data))
        assert result["a"].tolist() == [1, 2, 3]
