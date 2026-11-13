import numpy as np
import pytest
from app import App


class TestHoldoutSize:
    def test_equals_horizon_when_enough_data(self):
        # genug Daten: Testfenster = Horizont
        assert App._holdout_size(40, 12) == 12

    def test_capped_at_half_for_large_horizon(self):
        # zu großer Horizont: höchstens die Hälfte der Daten als Test
        assert App._holdout_size(40, 120) == 20

    def test_keeps_at_least_three_training_points(self):
        assert App._holdout_size(6, 12) == 3   # 6 - 3 = 3
        assert App._holdout_size(4, 10) == 1   # 4 - 3 = 1

    def test_no_holdout_when_minimal_data(self):
        assert App._holdout_size(3, 5) == 0

    def test_small_horizon_not_inflated(self):
        assert App._holdout_size(40, 4) == 4

    def test_train_plus_test_equals_n(self):
        for n, h in [(40, 12), (40, 120), (6, 12), (5, 3), (10, 4)]:
            ts = App._holdout_size(n, h)
            assert 0 <= ts <= n
            assert (n - ts) >= 3 or ts == 0


class TestEvaluateHoldout:
    def setup_method(self):
        self.app = App.__new__(App)  # ohne __init__ (kein Streamlit set_page_config)

    def test_perfect_trend_linear_has_zero_error(self):
        y = np.arange(1.0, 41.0)
        ts = App._holdout_size(len(y), 12)
        res = self.app._evaluate_holdout(y, len(y) - ts, ts, "Linearer Trend")
        assert res["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert res["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_returns_test_size(self):
        y = np.arange(1.0, 41.0)
        res = self.app._evaluate_holdout(y, 28, 12, "Naive Prognose")
        assert res["test_size"] == 12

    def test_naive_error_is_positive_on_trend(self):
        y = np.arange(1.0, 41.0)
        res = self.app._evaluate_holdout(y, 28, 12, "Naive Prognose")
        assert res["rmse"] > 0
        assert res["mae"] > 0

    def test_mase_present_and_positive(self):
        y = np.arange(1.0, 41.0)
        res = self.app._evaluate_holdout(y, 28, 12, "Naive Prognose")
        assert res["mase"] is not None
        assert res["mase"] > 0

    def test_mase_equals_mae_over_training_scale(self):
        y = np.arange(1.0, 41.0)  # konstante Schritte von 1 -> Skala = 1.0
        res = self.app._evaluate_holdout(y, 28, 12, "Naive Prognose")
        # Nenner = mittlerer absoluter Schritt der Trainingsdaten = 1.0 -> MASE == MAE
        assert res["mase"] == pytest.approx(res["mae"])

    def test_linear_beats_naive_on_trend(self):
        y = np.arange(1.0, 41.0)
        lin = self.app._evaluate_holdout(y, 28, 12, "Linearer Trend")
        assert lin["mase"] == pytest.approx(0.0, abs=1e-9)  # < 1: schlägt die Baseline

    def test_mase_none_when_training_flat(self):
        # flache Trainingsreihe -> Skala 0 -> MASE nicht definiert, kein Crash
        y = np.full(20, 5.0)
        res = self.app._evaluate_holdout(y, 12, 8, "Naive Prognose")
        assert res["mase"] is None

    def test_failure_returns_error_flag(self):
        # Holt-Winters mit viel zu wenig Trainingsdaten -> Fehler wird abgefangen
        y = np.arange(1.0, 9.0)
        res = self.app._evaluate_holdout(y, 5, 3, "Holt-Winters")
        assert res.get("error") is True
        assert res["test_size"] == 3
