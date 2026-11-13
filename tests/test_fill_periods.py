import numpy as np
import pandas as pd
import pytest
from app import App


def _cw_df(weeks, values):
    return pd.DataFrame({"Periode": [f"2026{w:02d}" for w in weeks], "Wert": values})


class TestFillMissingPeriods:
    def test_zero_fill_inserts_missing_weeks(self):
        # KW11,12 dann 15,16 -> KW13,14 fehlen
        df = _cw_df([11, 12, 15, 16], [1104, 1104, 336, 1008])
        y, _, _, n_inserted = App._prepare_series(df, "Wert", "Periode", "zero")
        assert n_inserted == 2
        assert list(y) == [1104.0, 1104.0, 0.0, 0.0, 336.0, 1008.0]

    def test_interpolate_fills_between_neighbors(self):
        df = _cw_df([11, 12, 15, 16], [1104, 1104, 336, 1008])
        y, _, _, n_inserted = App._prepare_series(df, "Wert", "Periode", "interpolate")
        assert n_inserted == 2
        # linear zwischen 1104 und 336 über 3 Schritte -> 848, 592
        assert list(y) == [1104.0, 1104.0, 848.0, 592.0, 336.0, 1008.0]

    def test_none_leaves_gaps(self):
        df = _cw_df([11, 12, 15, 16], [1104, 1104, 336, 1008])
        y, _, _, n_inserted = App._prepare_series(df, "Wert", "Periode", "none")
        assert n_inserted == 0
        assert len(y) == 4

    def test_no_gaps_inserts_nothing(self):
        df = _cw_df([11, 12, 13, 14], [10, 20, 30, 40])
        y, _, _, n_inserted = App._prepare_series(df, "Wert", "Periode", "zero")
        assert n_inserted == 0
        assert list(y) == [10.0, 20.0, 30.0, 40.0]

    def test_same_column_for_time_and_value_does_not_crash(self):
        # Regression: gleiche Spalte als Datum UND Wert -> ts[time_col] wird numerisch;
        # das Auffüllen darf nicht crashen (früher: TypeError float <= Timedelta)
        df = pd.DataFrame({"Periode": [202601, 202602, 202603, 202605]})
        y, _, _, n_inserted = App._prepare_series(df, "Periode", "Periode", "zero")
        assert n_inserted == 0
        assert len(y) == 4

    def test_irregular_spacing_not_filled(self):
        # überwiegend unregelmäßige Abstände -> kein Auffüllen (Schutz)
        df = pd.DataFrame({
            "Datum": ["2026-01-01", "2026-01-02", "2026-02-15", "2026-03-20"],
            "Wert":  [10, 20, 30, 40],
        })
        _, _, _, n_inserted = App._prepare_series(df, "Wert", "Datum", "zero")
        assert n_inserted == 0

    def test_fill_after_merging_duplicates(self):
        # doppelte Periode wird erst summiert, dann Lücke gefüllt
        df = pd.DataFrame({
            "Periode": ["202611", "202611", "202612", "202614"],  # KW11 doppelt, KW13 fehlt
            "Wert":    [100, 200, 50, 80],
        })
        y, n_invalid, n_merged, n_inserted = App._prepare_series(df, "Wert", "Periode", "zero")
        assert n_merged == 1          # zwei KW11 -> eine
        assert n_inserted == 1        # KW13 ergänzt
        assert list(y) == [300.0, 50.0, 0.0, 80.0]  # KW11=100+200, KW12=50, KW13=0, KW14=80


class TestTimeGrid:
    """Zeitraster Tag/Woche/Monat: Werte je Periode summieren und leere Perioden ergänzen."""

    @staticmethod
    def _sparse_daily():
        # sporadisch gebucht: nur 30 % der Abstände sind 1 Tag -> das automatische Auffüllen
        # (Regelmäßigkeitsprüfung) greift hier nicht, das feste Raster muss es trotzdem tun
        dates = pd.to_datetime(["2026-09-01", "2026-09-03", "2026-09-04", "2026-09-09",
                                "2026-09-11", "2026-09-17"])
        return pd.DataFrame({"Datum": dates, "Menge": [10, 20, 30, 40, 50, 60]})

    def test_auto_mode_skips_sparse_daily_data(self):
        # dokumentiert das alte Verhalten, das der Grund für das Raster ist
        _, _, _, n_inserted = App._prepare_series(self._sparse_daily(), "Menge", "Datum", "zero")
        assert n_inserted == 0

    def test_daily_grid_fills_every_missing_day(self):
        y, _, _, n_inserted = App._prepare_series(self._sparse_daily(), "Menge", "Datum", "zero", "D")
        assert len(y) == 17  # 01.09. bis 17.09.
        assert n_inserted == 11
        assert y.sum() == 210
        assert y[0] == 10 and y[1] == 0 and y[2] == 20

    def test_daily_grid_extends_to_file_range(self):
        rng = (pd.Timestamp("2026-08-30"), pd.Timestamp("2026-09-20"))
        y, _, _, _ = App._prepare_series(self._sparse_daily(), "Menge", "Datum", "zero", "D", rng)
        assert len(y) == 22
        assert y[:2].tolist() == [0.0, 0.0] and y[-3:].tolist() == [0.0, 0.0, 0.0]

    def test_weekly_grid_sums_per_calendar_week(self):
        # KW36 (31.08.–06.09.): 10+20+30, KW37: 40+50, KW38: 60
        y, _, n_merged, n_inserted = App._prepare_series(
            self._sparse_daily(), "Menge", "Datum", "zero", "W")
        assert y.tolist() == [60.0, 90.0, 60.0]
        assert n_merged == 3 and n_inserted == 0

    def test_weekly_grid_inserts_empty_weeks(self):
        df = pd.DataFrame({"Datum": pd.to_datetime(["2026-09-01", "2026-09-29"]), "Menge": [5, 7]})
        y, _, _, n_inserted = App._prepare_series(df, "Menge", "Datum", "zero", "W")
        assert y.tolist() == [5.0, 0.0, 0.0, 0.0, 7.0]
        assert n_inserted == 3

    def test_monthly_grid(self):
        df = pd.DataFrame({"Datum": pd.to_datetime(["2026-01-15", "2026-01-20", "2026-03-02"]),
                           "Menge": [1, 2, 4]})
        y, _, _, _ = App._prepare_series(df, "Menge", "Datum", "zero", "M")
        assert y.tolist() == [3.0, 0.0, 4.0]

    def test_calendar_week_values_with_weekly_grid(self):
        # KW-Werte (202611 ...) liegen schon auf Montagen -> Raster "Woche" ändert nichts
        df = _cw_df([11, 12, 15, 16], [1104, 1104, 336, 1008])
        y, _, _, n_inserted = App._prepare_series(df, "Wert", "Periode", "zero", "W")
        assert list(y) == [1104.0, 1104.0, 0.0, 0.0, 336.0, 1008.0]
        assert n_inserted == 2

    def test_interpolate_stays_within_data_range(self):
        rng = (pd.Timestamp("2026-08-01"), pd.Timestamp("2026-10-31"))
        df = pd.DataFrame({"Datum": pd.to_datetime(["2026-09-01", "2026-09-03"]), "Menge": [10, 30]})
        y, _, _, _ = App._prepare_series(df, "Menge", "Datum", "interpolate", "D", rng)
        assert y.tolist() == [10.0, 20.0, 30.0]

    def test_none_fill_only_aggregates(self):
        y, _, _, n_inserted = App._prepare_series(self._sparse_daily(), "Menge", "Datum", "none", "W")
        assert y.tolist() == [60.0, 90.0, 60.0] and n_inserted == 0

    def test_same_column_for_time_and_value_with_grid_does_not_crash(self):
        df = _cw_df([11, 12, 15], [1, 2, 3])
        y, _, _, _ = App._prepare_series(df, "Periode", "Periode", "zero", "W")
        assert len(y) == 3


class TestPartialPeriods:
    def test_file_from_tuesday_to_saturday(self):
        # 01.09.2026 = Dienstag, 31.10.2026 = Samstag
        rng = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-31"))
        assert App._partial_periods(rng, "W") == ["erste Periode", "letzte Periode"]
        assert App._partial_periods(rng, "M") == []
        assert App._partial_periods(rng, "D") == []

    def test_full_weeks(self):
        rng = (pd.Timestamp("2026-08-31"), pd.Timestamp("2026-11-01"))  # Mo bis So
        assert App._partial_periods(rng, "W") == []

    def test_no_range(self):
        assert App._partial_periods(None, "W") == []
