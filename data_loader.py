import io

import pandas as pd
import streamlit as st

_EXCEL_ENGINES = {"xlsx": "openpyxl", "xlsb": "pyxlsb"}


@st.cache_data
def _cached_load(data_bytes: bytes, name: str, sheet_name=0) -> pd.DataFrame:
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(data_bytes), header=None)
    engine = _EXCEL_ENGINES.get(ext)
    if engine:
        return pd.read_excel(io.BytesIO(data_bytes), header=None, engine=engine, sheet_name=sheet_name)
    return pd.DataFrame()


@st.cache_data
def _cached_sheet_names(data_bytes: bytes, name: str) -> list:
    ext = name.rsplit(".", 1)[-1].lower()
    engine = _EXCEL_ENGINES.get(ext)
    if not engine:
        return []
    return pd.ExcelFile(io.BytesIO(data_bytes), engine=engine).sheet_names


class DataLoader:
    SUPPORTED = {"xlsx", "xlsb", "csv"}

    @classmethod
    def sheet_names(cls, data_bytes: bytes, name: str) -> list:
        """Sheet names for Excel files; empty list for CSV or unsupported formats."""
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in _EXCEL_ENGINES:
            return []
        return _cached_sheet_names(data_bytes, name)

    @classmethod
    def load(cls, data_bytes: bytes, name: str, sheet_name=0) -> pd.DataFrame:
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in cls.SUPPORTED:
            raise ValueError(
                f"Format '.{ext}' nicht unterstützt. Bitte .xlsx, .xlsb oder .csv verwenden."
            )
        return _cached_load(data_bytes, name, sheet_name)

    @staticmethod
    def apply_header(df: pd.DataFrame, row: int) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.iloc[row]
        df = df.drop(index=range(row + 1)).reset_index(drop=True)
        df.columns = DataLoader._make_unique(df.columns)
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    @staticmethod
    def _make_unique(columns) -> list:
        seen, out = {}, []
        for c in columns:
            k   = str(c)
            cnt = seen.get(k, 0)
            out.append(k if cnt == 0 else f"{k}_{cnt}")
            seen[k] = cnt + 1
        return out
