import io

import pandas as pd


class Exporter:
    EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    @staticmethod
    def to_csv(df: pd.DataFrame, index: bool = True) -> bytes:
        return df.to_csv(index=index).encode("utf-8")

    @staticmethod
    def to_excel(df: pd.DataFrame, sheet: str = "Sheet1", index: bool = True) -> bytes:
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                df.to_excel(w, index=index, sheet_name=sheet)
        except Exception:
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, index=index, sheet_name=sheet)
        buf.seek(0)
        return buf.getvalue()
