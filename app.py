import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import DataLoader
from classifier import ColumnClassifier
from statistics import StatisticsAnalyzer
from forecaster import Forecaster, ForecastResult
from charts import ChartRenderer
from exporter import Exporter


class App:
    def __init__(self):
        st.set_page_config(page_title="Statistic-Tool", layout="wide")
        self._apply_compact_ui()

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        st.title("📊 Statistic-Tool")

        uploaded = st.file_uploader(
            "Datei auswählen (Excel oder CSV)", type=["xlsx", "xlsb", "csv"]
        )
        if not uploaded:
            return

        self._handle_upload(uploaded)

        if "df" not in st.session_state:
            return

        df = st.session_state["df"]
        self._section_data_preview(df)

        if not st.session_state.get("confirmed"):
            return

        self._section_statistics(df)
        self._section_pivot(df)
        self._section_forecast(df)

    # ── upload & data prep ────────────────────────────────────────────────────

    def _handle_upload(self, uploaded):
        if st.session_state.get("_fname") != uploaded.name:
            st.session_state.clear()
            st.session_state["_fname"] = uploaded.name
            st.session_state["_bytes"] = uploaded.read()

        data_bytes = st.session_state["_bytes"]
        sheets     = DataLoader.sheet_names(data_bytes, uploaded.name)

        sheet_name = 0
        if len(sheets) > 1:
            sheet_name = st.selectbox("Sheet auswählen", sheets, key="sel_sheet")
            if st.session_state.get("_sheet") != sheet_name:
                self._reset_session_keep(
                    _fname=uploaded.name, _bytes=data_bytes,
                    _sheet=sheet_name, sel_sheet=sheet_name,
                )

        try:
            df_raw = DataLoader.load(data_bytes, uploaded.name, sheet_name)
        except ValueError as e:
            st.error(str(e))
            return

        st.subheader("Rohdaten-Vorschau")
        st.dataframe(df_raw.head(8), use_container_width=True, height=200)

        header_row = st.number_input(
            "Header-Zeile (0-basiert)",
            min_value=0, max_value=max(len(df_raw) - 1, 0), value=0,
        )
        if st.button("Header bestätigen"):
            st.session_state["df"] = DataLoader.apply_header(df_raw, header_row)

    def _section_data_preview(self, df: pd.DataFrame):
        st.subheader("Bereinigte Daten")
        st.dataframe(df.head(8), use_container_width=True, height=220)
        st.multiselect("Spalten für die Analyse auswählen", list(df.columns), key="sel_cols")
        if st.button("Auswahl bestätigen"):
            st.session_state["confirmed"] = True

    # ── section 1: statistics ─────────────────────────────────────────────────

    def _section_statistics(self, df: pd.DataFrame):
        st.markdown("---")
        st.header("1 · Statistische Kennzahlen")

        num_cols = [c for c in st.session_state.get("sel_cols", [])
                    if ColumnClassifier.is_numeric(df[c])]
        if not num_cols:
            st.warning("Keine numerischen Spalten in der Auswahl.")
            return

        with st.form("stats_form"):
            c1, c2, c3, c4 = st.columns(4)
            checks = {
                "Mittelwert":         c1.checkbox("Mittelwert",          value=True),
                "Median":             c2.checkbox("Median",               value=True),
                "Standardabweichung": c3.checkbox("Standardabweichung",  value=True),
                "Minimum":            c4.checkbox("Minimum"),
                "Maximum":            c1.checkbox("Maximum"),
                "25%-Quantil":        c2.checkbox("25%-Quantil"),
                "75%-Quantil":        c3.checkbox("75%-Quantil"),
                "Anzahl (n)":         c4.checkbox("Anzahl (n)"),
            }
            if st.form_submit_button("Berechnen"):
                selected = [m for m, v in checks.items() if v]
                if selected:
                    analyzer = StatisticsAnalyzer(df, num_cols)
                    st.session_state["stats_df"] = analyzer.compute(selected)

        if "stats_df" not in st.session_state:
            return

        stats_df = st.session_state["stats_df"]
        st.dataframe(stats_df.T, use_container_width=True, height=280)

        analyzer = StatisticsAnalyzer(df, num_cols)
        self._show_invalid_values_caption(analyzer.invalid_counts())

        st.markdown("#### Balkendiagramme")
        for _, fig in analyzer.bar_charts(stats_df):
            st.pyplot(fig, use_container_width=False)

        st.markdown("#### Boxplots")
        for _, fig in analyzer.boxplots():
            st.pyplot(fig, use_container_width=False)

        self._export_buttons(stats_df, "statistik.csv", "statistik.xlsx", "Statistik")

    # ── section 2: pivot ──────────────────────────────────────────────────────

    def _section_pivot(self, df: pd.DataFrame):
        st.markdown("---")
        st.header("2 · Pivot")

        cc       = ColumnClassifier
        num_cols = cc.numeric_cols(df)
        cat_cols = cc.categorical_cols(df)

        if not num_cols:
            st.warning("Keine numerischen Spalten gefunden.")
            return
        if not cat_cols:
            st.warning("Keine kategorialen Spalten zum Gruppieren gefunden.")
            return

        g1, g2, g3, g4 = st.columns([2, 2, 1, 1])
        group_cols   = g1.multiselect("Gruppieren nach", cat_cols, default=cat_cols[:1], key="grp_cols")
        grp_val_cols = g2.multiselect("Wertspalten", num_cols, default=num_cols[:1], key="grp_vals")
        grp_agg      = g3.selectbox("Aggregation", StatisticsAnalyzer.AGG_NAMES, key="grp_agg")
        max_bars     = g4.number_input("Max. Balken im Diagramm", min_value=5, max_value=200,
                                       value=25, step=5, key="grp_max_bars")

        if not group_cols:
            st.info("Bitte mindestens eine Gruppierungsspalte wählen.")
            return
        if not grp_val_cols:
            st.info("Bitte mindestens eine Wertspalte wählen.")
            return

        if st.button("Pivot berechnen"):
            analyzer = StatisticsAnalyzer(df, grp_val_cols)
            st.session_state["grp_result"]    = analyzer.grouped_stats(group_cols, df, grp_agg)
            st.session_state["grp_agg_label"] = grp_agg
            st.session_state["grp_val_cols"]  = grp_val_cols

        if "grp_result" not in st.session_state:
            return

        grp_result  = st.session_state["grp_result"]
        grp_agg_lbl = st.session_state["grp_agg_label"]
        cached_vals = st.session_state["grp_val_cols"]

        st.dataframe(grp_result, use_container_width=True, height=350)
        self._show_invalid_values_caption(StatisticsAnalyzer(df, cached_vals).invalid_counts())

        total_groups = len(grp_result)
        max_bars     = int(st.session_state.get("grp_max_bars", 25))
        chart_data   = grp_result.head(max_bars)

        if total_groups > max_bars:
            st.info(
                f"Das Diagramm zeigt die Top {max_bars} von {total_groups} Gruppen. "
                "Erhöhe 'Max. Balken' oder exportiere die vollständige Tabelle."
            )

        fig = StatisticsAnalyzer(df, cached_vals).grouped_bar_chart(chart_data, grp_agg_lbl)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        self._export_buttons(grp_result, "pivot.csv", "pivot.xlsx", "Pivot")

    # ── section 3: forecast ───────────────────────────────────────────────────

    def _section_forecast(self, df: pd.DataFrame):
        st.markdown("---")
        st.header("3 · Prognose / Zeitreihenanalyse")

        cc       = ColumnClassifier
        num_cols = cc.numeric_cols(df)
        date_cls = cc.date_cols(df)

        if not num_cols:
            st.warning("Keine numerischen Spalten für Prognose.")
            return

        pr1, pr2 = st.columns(2)
        with pr1:
            st.markdown("**Zeitachse**")
            use_date = st.toggle("Datumsspalte verwenden",
                                 value=bool(date_cls), key="prog_use_date")
            if use_date:
                time_col = st.selectbox(
                    "Datumsspalte",
                    date_cls + [c for c in df.columns if c not in date_cls],
                    key="prog_tcol",
                )
            else:
                time_col = None
                st.caption("Zeitachse = Zeilenindex (1, 2, 3 …)")
            value_col = st.selectbox("Wertespalte (Y)", num_cols, key="prog_vcol")

        with pr2:
            st.markdown("**Methode & Horizont**")
            method  = st.selectbox(
                "Prognosemethode",
                ["Linearer Trend", "Gleitender Durchschnitt", "Exponentielle Glättung", "ARIMA", "Holt-Winters"],
                key="prog_method",
                help=(
                    "**Linearer Trend**: passt eine Gerade an die Daten an – gut bei gleichmäßigem Wachstum/Rückgang.\n\n"
                    "**Gleitender Durchschnitt**: mittelt die letzten Werte – glättet Schwankungen.\n\n"
                    "**Exponentielle Glättung**: gewichtet neuere Werte stärker als ältere.\n\n"
                    "**ARIMA**: statistisches Modell für Trends und Muster in Zeitreihen, gut bei komplexeren Verläufen.\n\n"
                    "**Holt-Winters**: wie ARIMA, aber für Daten mit wiederkehrender Saisonalität (z. B. monatliche Zyklen)."
                ),
            )
            horizon = st.slider("Prognosehorizont (Perioden)", 1, 120, 12, key="prog_h",
                                help="Wie viele zukünftige Perioden (z. B. Monate) vorhergesagt werden sollen.")
            if method == "Gleitender Durchschnitt":
                st.slider("Fenstergröße", 2, 50, 3, key="prog_maw",
                         help="Anzahl der letzten Perioden, die gemittelt werden. Größere Werte glätten stärker, "
                              "reagieren aber langsamer auf Veränderungen.")
            if method == "Exponentielle Glättung":
                st.slider("Glättungsparameter α", 0.01, 0.99, 0.3, 0.01, key="prog_alpha",
                         help="Bestimmt, wie stark neuere Werte gegenüber älteren gewichtet werden. "
                              "Nahe 1 = reagiert stark auf neue Werte, nahe 0 = glättet stark.")
            if method == "ARIMA":
                ac1, ac2, ac3 = st.columns(3)
                ac1.number_input("p (AR-Ordnung)", min_value=0, max_value=10, value=1, key="prog_arima_p",
                                 help="Anzahl vergangener Werte, die zur Vorhersage genutzt werden. "
                                      "Höhere Werte = komplexeres Modell.")
                ac2.number_input("d (Differenzierung)", min_value=0, max_value=2, value=1, key="prog_arima_d",
                                 help="Wie oft die Zeitreihe differenziert wird, um einen Trend zu entfernen. "
                                      "1 reicht für die meisten Trends aus.")
                ac3.number_input("q (MA-Ordnung)", min_value=0, max_value=10, value=1, key="prog_arima_q",
                                 help="Anzahl vergangener Prognosefehler, die berücksichtigt werden.")
            if method == "Holt-Winters":
                hw1, hw2 = st.columns(2)
                hw1.number_input("Periodenlänge (z.B. 12 = monatlich)", min_value=2, max_value=52, value=12, key="prog_hw_period",
                                 help="Länge eines vollständigen Saisonzyklus in Datenpunkten, "
                                      "z. B. 12 bei monatlichen Daten mit Jahresmuster, 7 bei täglichen Daten mit Wochenmuster.")
                hw2.selectbox("Saisonalität", ["additive", "multiplicative"], key="prog_hw_seasonal",
                             help="additive: Saisonschwankungen bleiben über die Zeit gleich groß.\n\n"
                                  "multiplicative: Saisonschwankungen wachsen mit dem Niveau der Werte.")

        y = self._prepare_series(df, value_col, time_col)
        dropped = len(df) - len(y)
        if dropped > 0:
            st.caption(
                f"{dropped} von {len(df)} Zeilen enthielten ungültige oder fehlende Werte "
                "in Datums-/Wertespalte und wurden für die Prognose ausgeschlossen."
            )
        if len(y) < 3:
            st.warning("Mindestens 3 Datenpunkte nötig.")
            return

        if not st.button("Prognose berechnen"):
            return

        try:
            result = self._run_forecast(Forecaster(y), method, horizon)
        except Exception as e:
            st.error(self._forecast_error_message(method))
            with st.expander("Technische Details"):
                st.code(str(e))
            return

        fig = ChartRenderer.forecast(result, value_col)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if result.r2 is not None:
            st.info(
                f"Steigung: **{result.slope:+.4f}** pro Periode · "
                f"R²: **{result.r2:.3f}** · RMSE: **{result.rmse:.2f}**"
            )
            st.caption(
                "R² (Bestimmtheitsmaß): wie gut die Gerade zu den historischen Daten passt "
                "(1,0 = perfekte Anpassung, 0 = kein Zusammenhang). "
                "RMSE: durchschnittlicher Vorhersagefehler in der Einheit der Werte – je kleiner, desto genauer."
            )

        prog_df = result.to_dataframe()
        st.markdown("#### Prognosetabelle")
        if result.ci_low is not None:
            st.caption(
                "KI = Konfidenzintervall: der Bereich, in dem der tatsächliche Wert "
                "mit 95%iger Wahrscheinlichkeit liegen wird."
            )
        st.dataframe(prog_df, use_container_width=True, height=240)
        self._export_buttons(prog_df, "prognose.csv", "prognose.xlsx", "Prognose", index=False)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _reset_session_keep(**keep):
        st.session_state.clear()
        for k, v in keep.items():
            st.session_state[k] = v

    @staticmethod
    def _show_invalid_values_caption(invalid_counts: dict):
        parts = [f"{col}: {n}" for col, n in invalid_counts.items() if n > 0]
        if parts:
            st.caption(
                "Ungültige oder fehlende Werte ausgeschlossen – " + ", ".join(parts)
            )

    @staticmethod
    def _forecast_error_message(method: str) -> str:
        hints = {
            "ARIMA": (
                "Die ARIMA-Berechnung ist fehlgeschlagen. "
                "Versuche kleinere Werte für p/d/q oder wähle eine andere Methode."
            ),
            "Holt-Winters": (
                "Die Holt-Winters-Berechnung ist fehlgeschlagen. "
                "Dafür werden mindestens zwei volle Perioden an Daten benötigt – "
                "reduziere die Periodenlänge oder wähle eine andere Methode."
            ),
        }
        return hints.get(
            method,
            f"Die Prognose ({method}) konnte mit den aktuellen Einstellungen nicht berechnet werden.",
        )

    def _run_forecast(self, forecaster: Forecaster, method: str, horizon: int) -> ForecastResult:
        if method == "Linearer Trend":
            return forecaster.linear(horizon)
        if method == "Gleitender Durchschnitt":
            return forecaster.moving_average(st.session_state.get("prog_maw", 3), horizon)
        if method == "ARIMA":
            p = int(st.session_state.get("prog_arima_p", 1))
            d = int(st.session_state.get("prog_arima_d", 1))
            q = int(st.session_state.get("prog_arima_q", 1))
            return forecaster.arima(p, d, q, horizon)
        if method == "Holt-Winters":
            period   = int(st.session_state.get("prog_hw_period", 12))
            seasonal = st.session_state.get("prog_hw_seasonal", "additive")
            return forecaster.holt_winters(period, seasonal, horizon)
        return forecaster.exp_smoothing(st.session_state.get("prog_alpha", 0.3), horizon)

    @staticmethod
    def _prepare_series(df: pd.DataFrame, value_col: str, time_col) -> np.ndarray:
        if time_col:
            ts = df[[time_col, value_col]].copy()
            # df[col] returns a DataFrame when duplicate column names exist — always extract a Series
            t_series = ts[time_col]
            if isinstance(t_series, pd.DataFrame):
                t_series = t_series.iloc[:, 0]
            v_series = ts[value_col]
            if isinstance(v_series, pd.DataFrame):
                v_series = v_series.iloc[:, 0]
            ts = pd.DataFrame({time_col: t_series.values, value_col: v_series.values})
            ts[time_col]  = pd.to_datetime(ts[time_col], errors="coerce")
            ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
            ts = ts.dropna().sort_values(time_col).reset_index(drop=True)
        else:
            ts = df[[value_col]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        return ts[value_col].to_numpy(dtype=float)

    @staticmethod
    def _export_buttons(df: pd.DataFrame, csv_name: str, xlsx_name: str,
                        sheet: str, index: bool = True):
        c1, c2 = st.columns(2)
        c1.download_button("CSV herunterladen",
                           data=Exporter.to_csv(df, index=index),
                           file_name=csv_name, mime="text/csv")
        c2.download_button("Excel herunterladen",
                           data=Exporter.to_excel(df, sheet=sheet, index=index),
                           file_name=xlsx_name, mime=Exporter.EXCEL_MIME)

    @staticmethod
    def _apply_compact_ui():
        if st.sidebar.checkbox("Compact UI", value=True):
            st.markdown("""
            <style>
              .block-container{padding-top:.75rem;padding-bottom:1rem}
              h1{font-size:1.6rem;margin:.4rem 0 .6rem 0}
              h2{font-size:1.3rem;margin:.4rem 0 .5rem 0}
              h3{font-size:1.1rem;margin:.3rem 0 .4rem 0}
              p,label,span,div{font-size:.95rem}
            </style>""", unsafe_allow_html=True)
            plt.rcParams.update({
                "figure.figsize": (7, 3.5), "figure.dpi": 110,
                "axes.titlesize": 10, "axes.labelsize": 10,
                "xtick.labelsize": 8,  "ytick.labelsize": 8,
                "legend.fontsize": 8,  "lines.linewidth": 1.2,
            })
