import pandas as pd


class ColumnClassifier:
    # ISO calendar-week format, e.g. 202653 = Jahr 2026, KW 53
    _CW_MIN_YEAR, _CW_MAX_YEAR = 1900, 2100

    @classmethod
    def _parse_calendar_week(cls, s: pd.Series) -> pd.Series:
        """Parse YYYYWW-style ISO calendar-week values (e.g. 202653) to the Monday of that week."""
        text = s.astype(str).str.strip()
        groups = text.str.extract(r"^(\d{4})(\d{2})$")
        year = pd.to_numeric(groups[0], errors="coerce")
        week = pd.to_numeric(groups[1], errors="coerce")
        valid = year.between(cls._CW_MIN_YEAR, cls._CW_MAX_YEAR) & week.between(1, 53)

        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        if valid.any():
            parsed.loc[valid] = pd.to_datetime(
                text[valid] + "1", format="%G%V%u", errors="coerce"
            )
        return parsed

    @classmethod
    def parse_dates(cls, s: pd.Series) -> pd.Series:
        """Parse a column to datetimes, recognizing ISO calendar weeks (YYYYWW) as well as normal dates."""
        cw = cls._parse_calendar_week(s)
        if len(s) and cw.notna().sum() / len(s) >= 0.7:
            return cw
        return pd.to_datetime(s, errors="coerce")

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
                # plain to_datetime on numbers misreads them as epoch timestamps,
                # so only the strict calendar-week pattern is accepted here
                cw = cls._parse_calendar_week(df[c])
                if len(df) and cw.notna().sum() / len(df) >= 0.7:
                    out.append(c)
                continue
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() / max(len(df), 1) >= 0.7:
                    out.append(c)
            except Exception:
                pass
        return out
