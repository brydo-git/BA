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
        st.set_page_config(page_title="Statistik- und Prognose Tool", layout="wide")
        self._apply_compact_ui()

    # ── entry point ───────────────────────────────────────────────────────────

    def run(self):
        st.title("Statistik- und Prognose Tool")

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
            plt.close(fig)

        st.markdown("#### Boxplots")
        for _, fig in analyzer.boxplots():
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

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
        cat_cols = cc.categorical_cols(df)

        if not num_cols:
            st.warning("Keine numerischen Spalten für Prognose.")
            return

        df_forecast = df
        group_cols  = []
        batch_mode  = False
        if cat_cols:
            group_cols = st.multiselect(
                "Gruppieren nach (optional)", cat_cols, key="prog_group_cols",
                help="Z.B. Materialnummer und Werk, um nur eine bestimmte Kombination oder alle auf einmal "
                     "zu prognostizieren, statt alle Zeilen vermischt in einer Zeitreihe.",
            )
            if group_cols:
                batch_mode = st.checkbox(
                    "Alle Kombinationen auf einmal prognostizieren", key="prog_batch_mode",
                    help="Berechnet für jede Kombination eine eigene Prognose, statt nur für eine ausgewählte.",
                )
                if not batch_mode:
                    df_forecast = self._select_forecast_group(df, group_cols)
                    st.caption(f"{len(df_forecast)} von {len(df)} Zeilen für diese Kombination.")

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
                fill_label = st.selectbox(
                    "Fehlende Perioden auffüllen",
                    ["Mit 0 auffüllen", "Interpolieren", "Nicht auffüllen"],
                    key="prog_fill",
                    help="Ergänzt ausgelassene Perioden (z. B. fehlende Kalenderwochen) auf ein "
                         "lückenloses Raster.\n\n"
                         "**Mit 0 auffüllen**: fehlende Woche = kein Wert/keine Lieferung (Menge 0).\n\n"
                         "**Interpolieren**: fehlende Woche = Zwischenwert aus den Nachbarwochen.\n\n"
                         "**Nicht auffüllen**: Lücken bleiben – fehlende Perioden werden übersprungen "
                         "(wichtig v. a. für Holt-Winters, das ein regelmäßiges Raster braucht).",
                )
            else:
                time_col = None
                fill_label = "Nicht auffüllen"
                st.caption("Zeitachse = Zeilenindex (1, 2, 3 …)")
            # Datumsspalte nicht als Wertespalte anbieten (sonst Datum == Y)
            value_choices = [c for c in num_cols if c != time_col] if time_col else num_cols
            if not value_choices:
                st.warning("Keine numerische Wertespalte außer der Datumsspalte vorhanden.")
                return
            value_col = st.selectbox("Wertespalte (Y)", value_choices, key="prog_vcol")

        fill_mode = {"Mit 0 auffüllen": "zero", "Interpolieren": "interpolate"}.get(fill_label, "none")

        with pr2:
            st.markdown("**Methode & Horizont**")
            method  = st.selectbox(
                "Prognosemethode",
                ["Naive Prognose", "Linearer Trend", "Gleitender Durchschnitt",
                 "Exponentielle Glättung", "ARIMA", "Holt-Winters"],
                key="prog_method",
                help=(
                    "**Naive Prognose**: schreibt den letzten beobachteten Wert unverändert fort. "
                    "Dient als Referenzmodell (einfache Vergleichsbasis) – eine gute Methode sollte sie schlagen.\n\n"
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

        if batch_mode:
            self._section_forecast_batch(df, group_cols, value_col, time_col, method, horizon, fill_mode)
            return

        y, n_invalid, n_merged, n_inserted = self._prepare_series(
            df_forecast, value_col, time_col, fill_mode
        )
        notes = []
        if n_invalid > 0:
            notes.append(
                f"{n_invalid} Zeile(n) mit ungültigen oder fehlenden Werten in Datums-/Wertespalte "
                "ausgeschlossen"
            )
        if n_merged > 0:
            notes.append(f"{n_merged} Zeile(n) mit gleicher Periode zusammengefasst (summiert)")
        if n_inserted > 0:
            fill_word = "mit 0 aufgefüllt" if fill_mode == "zero" else "interpoliert"
            notes.append(f"{n_inserted} fehlende Periode(n) ergänzt und {fill_word}")
        if notes:
            st.caption(
                " · ".join(notes) + f" — {len(y)} Datenpunkte für die Prognose."
            )
        if len(y) < 3:
            st.warning("Mindestens 3 Datenpunkte nötig.")
            return

        n          = len(y)
        test_size  = self._holdout_size(n, horizon)
        train_size = n - test_size

        st.markdown("#### Datenaufteilung")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Datenpunkte", n)
        d2.metric("Trainingsdaten", train_size)
        d3.metric("Testdaten", test_size)
        d4.metric("Letzter Trainingswert (Eval.)",
                  f"{y[train_size - 1]:g}" if test_size > 0 else "–")
        d5.metric("Letzter Gesamtwert (Zukunft)", f"{y[-1]:g}")
        if test_size == 0:
            st.caption("Zu wenige Datenpunkte für eine Holdout-Bewertung – es wird nur die "
                       "Zukunftsprognose auf allen Daten berechnet.")
        elif test_size < horizon:
            st.caption(f"Testfenster auf {test_size} Perioden begrenzt (Horizont {horizon}), "
                       "damit genug Trainingsdaten bleiben.")

        if st.button("Prognose berechnen"):
            st.session_state.pop("forecast_eval", None)
            try:
                st.session_state["forecast_result"]   = self._run_forecast(Forecaster(y), method, horizon)
                st.session_state["forecast_value_col"] = value_col
            except Exception as e:
                st.session_state.pop("forecast_result", None)
                st.error(self._forecast_error_message(method))
                with st.expander("Technische Details"):
                    st.code(str(e))
            else:
                if test_size > 0:
                    st.session_state["forecast_eval"] = self._evaluate_holdout(
                        y, train_size, test_size, method
                    )

        if "forecast_result" not in st.session_state:
            return

        result    = st.session_state["forecast_result"]
        value_col = st.session_state["forecast_value_col"]

        fig = ChartRenderer.forecast(result, value_col)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        eval_info = st.session_state.get("forecast_eval")
        if eval_info:
            if eval_info.get("error"):
                st.warning(
                    "Holdout-Bewertung für diese Methode/Einstellungen nicht möglich "
                    "(zu wenige Trainingsdaten für das Modell)."
                )
            else:
                line = f"RMSE: **{eval_info['rmse']:.2f}** · MAE: **{eval_info['mae']:.2f}**"
                if eval_info.get("mase") is not None:
                    line += f" · MASE: **{eval_info['mase']:.2f}**"
                st.success(
                    f"**Bewertung auf Testdaten (out-of-sample, {eval_info['test_size']} Perioden)** — "
                    + line
                )
                st.caption(
                    "Das Modell wurde nur auf den Trainingsdaten trainiert und gegen die "
                    "zurückgehaltenen Testperioden geprüft. Dieser Fehler schätzt die echte "
                    "Prognosegüte und ist zwischen den Methoden fair vergleichbar. "
                    "MASE ist skalenfrei: **< 1 = besser als die naive Prognose**, > 1 = schlechter."
                )

        metrics = []
        if result.rmse is not None:
            metrics.append(f"RMSE: **{result.rmse:.2f}**")
        if result.mae is not None:
            metrics.append(f"MAE: **{result.mae:.2f}**")
        if result.r2 is not None:
            metrics.append(f"R²: **{result.r2:.3f}**")
        if result.slope is not None:
            metrics.append(f"Steigung: **{result.slope:+.4f}** pro Periode")

        if metrics:
            st.info("In-Sample-Fit (alle Daten) — " + " · ".join(metrics))
            st.caption(
                "RMSE und MAE hier messen, wie gut das Modell zu den **historischen** Daten passt "
                "(In-Sample-Fit), jeweils in der Einheit der Werte – je kleiner, desto besser. "
                "RMSE bestraft große Ausreißer stärker, MAE ist der durchschnittliche absolute Fehler. "
                "Für den fairen Methodenvergleich ist die Bewertung auf Testdaten (oben) aussagekräftiger."
            )
            if result.r2 is not None:
                st.caption(
                    "R² (Bestimmtheitsmaß): wie gut die Gerade zu den historischen Daten passt "
                    "(1,0 = perfekte Anpassung, 0 = kein Zusammenhang)."
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

    def _section_forecast_batch(self, df: pd.DataFrame, group_cols: list, value_col: str,
                                 time_col, method: str, horizon: int, fill_mode: str = "none"):
        combos = df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
        st.info(f"{len(combos)} Kombinationen gefunden.")
        if len(combos) > 50:
            st.caption(
                "Bei vielen Kombinationen und ARIMA/Holt-Winters kann die Berechnung etwas dauern."
            )

        if st.button("Prognose für alle Kombinationen berechnen"):
            results, errors, skipped = {}, [], []
            rows = []
            progress = st.progress(0.0)
            for i, combo in enumerate(combos.itertuples(index=False)):
                label = " | ".join(str(v) for v in combo)
                mask = pd.Series(True, index=df.index)
                for col, val in zip(group_cols, combo):
                    mask &= df[col] == val
                subset = df[mask]

                y, _, _, _ = self._prepare_series(subset, value_col, time_col, fill_mode)
                if len(y) < 3:
                    skipped.append(label)
                else:
                    try:
                        result = self._run_forecast(Forecaster(y), method, horizon)
                    except Exception as e:
                        errors.append((label, str(e)))
                    else:
                        results[label] = result
                        prog_df = result.to_dataframe()
                        for col, val in zip(reversed(group_cols), reversed(combo)):
                            prog_df.insert(0, col, val)
                        rows.append(prog_df)
                progress.progress((i + 1) / len(combos))
            progress.empty()

            st.session_state["forecast_batch"] = {
                "results":   results,
                "errors":    errors,
                "skipped":   skipped,
                "combined":  pd.concat(rows, ignore_index=True) if rows else None,
                "value_col": value_col,
                "method":    method,
                "total":     len(combos),
            }

        batch = st.session_state.get("forecast_batch")
        if not batch:
            return

        st.success(
            f"{len(batch['results'])} von {batch['total']} Kombinationen erfolgreich prognostiziert "
            f"({len(batch['skipped'])} übersprungen wegen zu weniger Daten, {len(batch['errors'])} fehlgeschlagen)."
        )
        if batch["skipped"]:
            with st.expander(f"Übersprungen wegen zu weniger Daten ({len(batch['skipped'])})"):
                st.write(", ".join(batch["skipped"]))
        if batch["errors"]:
            with st.expander(f"Fehlgeschlagen ({len(batch['errors'])})"):
                st.caption(self._forecast_error_message(batch["method"]))
                for label, msg in batch["errors"]:
                    st.write(f"**{label}**")
                    st.code(msg)

        if batch["combined"] is None:
            return

        st.markdown("#### Kombinierte Prognosetabelle")
        st.dataframe(batch["combined"], use_container_width=True, height=320)
        self._export_buttons(batch["combined"], "prognose_batch.csv", "prognose_batch.xlsx",
                             "Prognose_Batch", index=False)

        st.markdown("#### Einzelne Kombination ansehen")
        chosen = st.selectbox("Kombination", list(batch["results"].keys()), key="prog_batch_view")
        fig = ChartRenderer.forecast(batch["results"][chosen], batch["value_col"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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

    @staticmethod
    def _holdout_size(n: int, horizon: int) -> int:
        """Größe des Testfensters für die Holdout-Bewertung: = Horizont, aber gedeckelt,
        damit mindestens 3 Trainingspunkte und höchstens die Hälfte der Daten als Test bleiben."""
        return max(0, min(horizon, n - 3, n // 2))

    def _evaluate_holdout(self, y: np.ndarray, train_size: int, test_size: int, method: str) -> dict:
        """Trainiert nur auf den ersten train_size Punkten, prognostiziert das Testfenster und
        vergleicht mit den zurückgehaltenen echten Werten (out-of-sample RMSE/MAE/MASE).

        MASE = Test-MAE geteilt durch den MAE einer naiven Ein-Schritt-Prognose auf den
        Trainingsdaten (mittlerer absoluter Schritt). Skalenfrei: <1 = besser als naiv.
        """
        y_train, y_test = y[:train_size], y[train_size:]
        try:
            eval_res = self._run_forecast(Forecaster(y_train), method, test_size)
            y_pred   = np.asarray(eval_res.y_fut, dtype=float)[:test_size]
            resid    = y_test - y_pred
            mae      = float(np.mean(np.abs(resid)))
            scale    = float(np.mean(np.abs(np.diff(y_train))))  # naive Ein-Schritt-MAE (Training)
            return {
                "rmse":      float(np.sqrt(np.mean(resid ** 2))),
                "mae":       mae,
                "mase":      (mae / scale) if scale > 0 else None,
                "test_size": test_size,
            }
        except Exception:
            return {"error": True, "test_size": test_size}

    def _run_forecast(self, forecaster: Forecaster, method: str, horizon: int) -> ForecastResult:
        if method == "Naive Prognose":
            return forecaster.naive(horizon)
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
    def _select_forecast_group(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        """Let the user pick one specific combination of group_cols; returns the matching rows."""
        combos = df[group_cols].drop_duplicates().sort_values(group_cols).reset_index(drop=True)
        labels = [" | ".join(str(v) for v in row) for row in combos.itertuples(index=False)]

        # key depends on group_cols so switching the grouping columns doesn't reuse a stale selection
        key    = "prog_group_combo_" + "_".join(group_cols)
        choice = st.selectbox("Kombination auswählen", labels, key=key)
        selected = combos.iloc[labels.index(choice)]

        mask = pd.Series(True, index=df.index)
        for col in group_cols:
            mask &= df[col] == selected[col]
        return df[mask]

    @staticmethod
    def _fill_missing_periods(ts: pd.DataFrame, time_col: str, value_col: str, mode: str):
        """Ergänzt fehlende Perioden auf ein lückenloses Raster (häufigster Zeitabstand).

        mode: 'zero' (mit 0 füllen) oder 'interpolate' (linear). Rückgabe: (ts, n_inserted).
        Wird nur angewandt, wenn die vorhandenen Perioden gleichmäßig auf dem Raster liegen
        (z. B. Kalenderwochen); sonst bleibt ts unverändert.
        """
        if mode not in ("zero", "interpolate") or len(ts) < 3:
            return ts, 0
        if not pd.api.types.is_datetime64_any_dtype(ts[time_col]):
            return ts, 0  # ohne echte Datumsachse lässt sich kein Periodenraster bilden
        diffs = ts[time_col].diff().dropna()
        if diffs.empty:
            return ts, 0
        step = diffs.mode().iloc[0]  # häufigster Abstand = reguläre Periodenlänge
        if step <= pd.Timedelta(0):
            return ts, 0
        # nur auffüllen, wenn die Reihe überwiegend regelmäßig ist (sonst sind es keine
        # echten "Lücken", sondern unregelmäßige Zeitpunkte, die man nicht sinnvoll auffüllt)
        if (diffs == step).mean() < 0.5:
            return ts, 0
        full = pd.date_range(ts[time_col].min(), ts[time_col].max(), freq=step)
        # nur auffüllen, wenn jede vorhandene Periode exakt auf dem Raster liegt
        if len(full) <= len(ts) or not ts[time_col].isin(full).all():
            return ts, 0
        n_inserted = len(full) - len(ts)
        ts = ts.set_index(time_col).reindex(full)
        if mode == "zero":
            ts[value_col] = ts[value_col].fillna(0.0)
        else:
            ts[value_col] = ts[value_col].interpolate(method="linear")
        ts = ts.rename_axis(time_col).reset_index()
        return ts, n_inserted

    @staticmethod
    def _prepare_series(df: pd.DataFrame, value_col: str, time_col, fill_mode: str = "none"):
        """Bereitet die Zeitreihe auf.

        Rückgabe: (y, n_invalid, n_merged, n_inserted)
          n_invalid  = Zeilen mit ungültigem/fehlendem Datum oder Wert (ausgeschlossen)
          n_merged   = gültige Zeilen, die wegen gleicher Periode zusammengefasst (summiert) wurden
          n_inserted = fehlende Perioden, die aufs Raster ergänzt wurden (nur bei fill_mode != 'none')
        """
        n_rows     = len(df)
        n_inserted = 0
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
            ts[time_col]  = ColumnClassifier.parse_dates(ts[time_col])
            ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
            ts = ts.dropna().sort_values(time_col).reset_index(drop=True)
            n_valid = len(ts)  # nach Ausschluss ungültiger Zeilen, vor dem Zusammenfassen
            if ts[time_col].duplicated().any():
                # multiple rows for the same period (e.g. several deliveries in one week) — sum them
                ts = ts.groupby(time_col, as_index=False)[value_col].sum().sort_values(time_col).reset_index(drop=True)
            n_merged = n_valid - len(ts)
            ts, n_inserted = App._fill_missing_periods(ts, time_col, value_col, fill_mode)
        else:
            ts = df[[value_col]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            n_valid  = len(ts)
            n_merged = 0
        n_invalid = n_rows - n_valid
        return ts[value_col].to_numpy(dtype=float), n_invalid, n_merged, n_inserted

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
