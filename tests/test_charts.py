import numpy as np
import pytest
import matplotlib.pyplot as plt
from forecaster import ForecastResult
from charts import ChartRenderer


@pytest.fixture
def linear_result():
    return ForecastResult(
        method="Linearer Trend",
        y_hist=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        x_num=np.arange(5, dtype=float),
        x_fut=np.arange(5, 8, dtype=float),
        y_fut=np.array([6.0, 7.0, 8.0]),
        y_fitted=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        r2=1.0, rmse=0.0, slope=1.0,
    )


@pytest.fixture
def ma_result():
    return ForecastResult(
        method="Gleitender Durchschnitt (Fenster=3)",
        y_hist=np.array([1.0, 2.0, 3.0]),
        x_num=np.arange(3, dtype=float),
        x_fut=np.arange(3, 5, dtype=float),
        y_fut=np.array([2.0, 2.0]),
        y_fitted=np.array([1.0, 1.5, 2.0]),
    )


class TestForecastChart:
    def test_returns_matplotlib_figure(self, linear_result):
        fig = ChartRenderer.forecast(linear_result, "sales")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_contains_method_name(self, linear_result):
        fig = ChartRenderer.forecast(linear_result, "sales")
        assert "Linearer Trend" in fig.axes[0].get_title()
        plt.close(fig)

    def test_title_contains_column_name(self, linear_result):
        fig = ChartRenderer.forecast(linear_result, "my_column")
        assert "my_column" in fig.axes[0].get_title()
        plt.close(fig)

    def test_title_contains_r2_when_present(self, linear_result):
        fig = ChartRenderer.forecast(linear_result, "sales")
        assert "R²" in fig.axes[0].get_title()
        plt.close(fig)

    def test_title_has_no_r2_when_absent(self, ma_result):
        fig = ChartRenderer.forecast(ma_result, "sales")
        assert "R²" not in fig.axes[0].get_title()
        plt.close(fig)

    def test_works_with_ci(self, linear_result):
        linear_result.ci_low  = linear_result.y_fut - 1.0
        linear_result.ci_high = linear_result.y_fut + 1.0
        fig = ChartRenderer.forecast(linear_result, "qty")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_without_fitted_line(self):
        result = ForecastResult(
            method="Flat", y_hist=np.array([1.0, 2.0]),
            x_num=np.arange(2, dtype=float),
            x_fut=np.array([2.0]),
            y_fut=np.array([2.0]),
        )
        fig = ChartRenderer.forecast(result, "val")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
