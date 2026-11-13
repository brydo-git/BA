import numpy as np
import pytest
from forecaster import Forecaster, ForecastResult


@pytest.fixture
def linear_y():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def flat_y():
    return np.array([5.0, 5.0, 5.0, 5.0, 5.0])


class TestLinear:
    def test_perfect_trend_slope_is_one(self, linear_y):
        result = Forecaster(linear_y).linear(1)
        assert result.slope == pytest.approx(1.0)

    def test_perfect_trend_r2_is_one(self, linear_y):
        result = Forecaster(linear_y).linear(1)
        assert result.r2 == pytest.approx(1.0)

    def test_forecast_continues_sequence(self, linear_y):
        result = Forecaster(linear_y).linear(1)
        assert result.y_fut[0] == pytest.approx(6.0)

    def test_forecast_length_matches_steps(self, linear_y):
        result = Forecaster(linear_y).linear(7)
        assert len(result.y_fut) == 7
        assert len(result.x_fut) == 7

    def test_fitted_length_matches_history(self, linear_y):
        result = Forecaster(linear_y).linear(3)
        assert len(result.y_fitted) == len(linear_y)

    def test_ci_bounds_present(self, linear_y):
        result = Forecaster(linear_y).linear(3)
        assert result.ci_low is not None
        assert result.ci_high is not None

    def test_ci_lower_le_forecast(self, linear_y):
        result = Forecaster(linear_y).linear(5)
        assert np.all(result.ci_low <= result.y_fut)

    def test_ci_upper_ge_forecast(self, linear_y):
        result = Forecaster(linear_y).linear(5)
        assert np.all(result.ci_high >= result.y_fut)

    def test_flat_line_slope_is_zero(self, flat_y):
        result = Forecaster(flat_y).linear(1)
        assert result.slope == pytest.approx(0.0, abs=1e-10)

    def test_method_label(self, linear_y):
        result = Forecaster(linear_y).linear(1)
        assert result.method == "Linearer Trend"


class TestMovingAverage:
    def test_forecast_is_mean_of_last_n(self):
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        result = Forecaster(y).moving_average(window=2, steps=1)
        assert result.y_fut[0] == pytest.approx(15.0)

    def test_forecast_is_constant(self, flat_y):
        result = Forecaster(flat_y).moving_average(window=3, steps=4)
        assert np.all(result.y_fut == result.y_fut[0])

    def test_forecast_length(self, flat_y):
        result = Forecaster(flat_y).moving_average(window=3, steps=6)
        assert len(result.y_fut) == 6

    def test_window_larger_than_data_uses_all(self):
        y = np.array([2.0, 4.0])
        result = Forecaster(y).moving_average(window=100, steps=1)
        assert result.y_fut[0] == pytest.approx(3.0)

    def test_rolling_length_matches_history(self, flat_y):
        result = Forecaster(flat_y).moving_average(window=3, steps=1)
        assert len(result.y_fitted) == len(flat_y)

    def test_method_label_includes_window(self):
        result = Forecaster(np.array([1.0, 2.0, 3.0])).moving_average(window=5, steps=1)
        assert "5" in result.method


class TestExpSmoothing:
    def test_alpha_one_tracks_last_value(self):
        y = np.array([1.0, 2.0, 3.0, 99.0])
        result = Forecaster(y).exp_smoothing(alpha=1.0, steps=1)
        assert result.y_fut[0] == pytest.approx(99.0)

    def test_alpha_zero_stays_at_first_value(self):
        y = np.array([5.0, 100.0, 200.0])
        result = Forecaster(y).exp_smoothing(alpha=0.0, steps=1)
        assert result.y_fut[0] == pytest.approx(5.0)

    def test_forecast_is_constant(self, flat_y):
        result = Forecaster(flat_y).exp_smoothing(alpha=0.5, steps=5)
        assert np.all(result.y_fut == result.y_fut[0])

    def test_forecast_length(self, flat_y):
        result = Forecaster(flat_y).exp_smoothing(alpha=0.3, steps=8)
        assert len(result.y_fut) == 8

    def test_smoothed_starts_at_first_y(self):
        y = np.array([10.0, 20.0, 30.0])
        result = Forecaster(y).exp_smoothing(alpha=0.5, steps=1)
        assert result.y_fitted[0] == pytest.approx(10.0)

    def test_smoothed_length_matches_history(self, flat_y):
        result = Forecaster(flat_y).exp_smoothing(alpha=0.3, steps=1)
        assert len(result.y_fitted) == len(flat_y)

    def test_method_label_includes_alpha(self):
        result = Forecaster(np.array([1.0, 2.0, 3.0])).exp_smoothing(alpha=0.3, steps=1)
        assert "0.3" in result.method


class TestNaive:
    def test_forecast_is_last_value(self):
        y = np.array([1.0, 2.0, 3.0, 42.0])
        result = Forecaster(y).naive(1)
        assert result.y_fut[0] == pytest.approx(42.0)

    def test_forecast_is_constant(self):
        y = np.array([1.0, 2.0, 3.0, 42.0])
        result = Forecaster(y).naive(5)
        assert np.all(result.y_fut == 42.0)

    def test_forecast_length(self, flat_y):
        result = Forecaster(flat_y).naive(6)
        assert len(result.y_fut) == 6

    def test_fitted_length_matches_history(self, linear_y):
        result = Forecaster(linear_y).naive(3)
        assert len(result.y_fitted) == len(linear_y)

    def test_fitted_is_previous_value(self):
        y = np.array([10.0, 20.0, 30.0])
        result = Forecaster(y).naive(1)
        # jeder Punkt wird durch seinen Vorgänger geschätzt; erster Punkt = sich selbst
        assert list(result.y_fitted) == [10.0, 10.0, 20.0]

    def test_flat_series_has_zero_error(self, flat_y):
        result = Forecaster(flat_y).naive(1)
        assert result.rmse == pytest.approx(0.0)
        assert result.mae == pytest.approx(0.0)

    def test_populates_rmse_and_mae(self):
        y = np.array([1.0, 3.0, 2.0, 8.0])
        result = Forecaster(y).naive(1)
        assert result.rmse is not None
        assert result.mae is not None

    def test_method_label(self, linear_y):
        result = Forecaster(linear_y).naive(1)
        assert result.method == "Naive Prognose"


class TestFitMetrics:
    def test_linear_perfect_fit_has_zero_error(self, linear_y):
        result = Forecaster(linear_y).linear(1)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        assert result.mae == pytest.approx(0.0, abs=1e-9)

    def test_linear_populates_rmse_and_mae(self):
        y = np.array([1.0, 2.0, 2.0, 4.0, 5.0])  # nicht perfekt linear
        result = Forecaster(y).linear(1)
        assert result.rmse is not None and result.rmse > 0
        assert result.mae is not None and result.mae > 0

    def test_moving_average_populates_rmse_and_mae(self):
        y = np.array([1.0, 3.0, 2.0, 8.0, 5.0])
        result = Forecaster(y).moving_average(window=2, steps=1)
        assert result.rmse is not None
        assert result.mae is not None

    def test_exp_smoothing_populates_rmse_and_mae(self):
        y = np.array([1.0, 3.0, 2.0, 8.0, 5.0])
        result = Forecaster(y).exp_smoothing(alpha=0.3, steps=1)
        assert result.rmse is not None
        assert result.mae is not None

    def test_mae_never_exceeds_rmse(self):
        # mathematische Invariante: MAE <= RMSE
        y = np.array([2.0, 5.0, 1.0, 9.0, 4.0, 7.0])
        for result in (
            Forecaster(y).linear(1),
            Forecaster(y).moving_average(window=3, steps=1),
            Forecaster(y).exp_smoothing(alpha=0.4, steps=1),
        ):
            assert result.mae <= result.rmse + 1e-9

    def test_fit_errors_known_values(self):
        y      = np.array([2.0, 4.0, 6.0])
        fitted = np.array([1.0, 4.0, 8.0])  # Fehler: 1, 0, 2
        rmse, mae = Forecaster._fit_errors(y, fitted)
        assert mae == pytest.approx((1 + 0 + 2) / 3)
        assert rmse == pytest.approx(np.sqrt((1 + 0 + 4) / 3))


class TestForecastResult:
    def test_to_dataframe_basic_columns(self):
        r = ForecastResult(
            method="test", y_hist=np.array([1.0, 2.0, 3.0]),
            x_num=np.arange(3, dtype=float),
            x_fut=np.arange(3, 6, dtype=float),
            y_fut=np.array([4.0, 5.0, 6.0]),
        )
        df = r.to_dataframe()
        assert list(df.columns) == ["Periode", "Prognose"]

    def test_to_dataframe_length(self):
        r = ForecastResult(
            method="test", y_hist=np.array([1.0, 2.0]),
            x_num=np.arange(2, dtype=float),
            x_fut=np.arange(2, 7, dtype=float),
            y_fut=np.full(5, 3.0),
        )
        assert len(r.to_dataframe()) == 5

    def test_to_dataframe_periode_starts_after_history(self):
        r = ForecastResult(
            method="test", y_hist=np.array([1.0, 2.0, 3.0]),
            x_num=np.arange(3, dtype=float),
            x_fut=np.arange(3, 5, dtype=float),
            y_fut=np.array([4.0, 5.0]),
        )
        assert r.to_dataframe()["Periode"].iloc[0] == 4

    def test_to_dataframe_includes_ci_when_present(self):
        r = ForecastResult(
            method="test", y_hist=np.array([1.0, 2.0]),
            x_num=np.arange(2, dtype=float),
            x_fut=np.array([2.0, 3.0]),
            y_fut=np.array([3.0, 4.0]),
            ci_low=np.array([2.0, 3.0]),
            ci_high=np.array([4.0, 5.0]),
        )
        df = r.to_dataframe()
        assert "KI unten (95%)" in df.columns
        assert "KI oben  (95%)" in df.columns

    def test_to_dataframe_no_ci_when_absent(self):
        r = ForecastResult(
            method="test", y_hist=np.array([1.0]),
            x_num=np.array([0.0]),
            x_fut=np.array([1.0]),
            y_fut=np.array([2.0]),
        )
        df = r.to_dataframe()
        assert "KI unten (95%)" not in df.columns
