import pandas as pd


class ColumnClassifier:
    @staticmethod
    def is_numeric(s: pd.Series) -> bool:
        try:
            return bool(pd.to_numeric(s, errors="coerce").notna().sum() > 0)
        except Exception:
            return False

    @staticmethod
    def is_categorical(s: pd.Series, max_abs: int = 100, max_ratio: float = 0.05) -> bool:
        try:
            u = s.astype(str).nunique(dropna=True)
            return u <= max_abs or (len(s) > 0 and u / len(s) <= max_ratio)
        except Exception:
            return True

    @staticmethod
    def nunique(s: pd.Series) -> int:
        try:
            return s.astype(str).nunique(dropna=True)
        except Exception:
            return s.nunique(dropna=True)

    @classmethod
    def numeric_cols(cls, df: pd.DataFrame) -> list:
        return [c for c in df.columns if cls.is_numeric(df[c])]

    @classmethod
    def categorical_cols(cls, df: pd.DataFrame) -> list:
        return [c for c in df.columns
                if not cls.is_numeric(df[c]) or cls.is_categorical(df[c])]

    @classmethod
    def date_cols(cls, df: pd.DataFrame) -> list:
        out = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() / max(len(df), 1) >= 0.7:
                    out.append(c)
            except Exception:
                pass
        return out
