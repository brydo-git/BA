import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclasses.dataclass
class ForecastResult:
    method:   str
    y_hist:   np.ndarray
    x_num:    np.ndarray
    x_fut:    np.ndarray
    y_fut:    np.ndarray
    y_fitted: Optional[np.ndarray] = None
    ci_low:   Optional[np.ndarray] = None
    ci_high:  Optional[np.ndarray] = None
    r2:       Optional[float] = None
    rmse:     Optional[float] = None
    mae:      Optional[float] = None
    slope:    Optional[float] = None

    def to_dataframe(self) -> pd.DataFrame:
        n = len(self.y_hist)
        cols = {
            "Periode":  np.arange(n + 1, n + 1 + len(self.y_fut)),
            "Prognose": np.round(self.y_fut, 4),
        }
        if self.ci_low is not None:
            cols["KI unten (95%)"] = np.round(self.ci_low, 4)
            cols["KI oben  (95%)"] = np.round(self.ci_high, 4)
        return pd.DataFrame(cols)


class Forecaster:
    def __init__(self, y: np.ndarray):
        self._y = y
        self._n = len(y)
        self._x = np.arange(self._n, dtype=float)

    @staticmethod
    def _fit_errors(y: np.ndarray, fitted: np.ndarray) -> tuple:
        """In-Sample-Fehler des Modells gegenüber den historischen Daten."""
        resid = y - fitted
        rmse  = float(np.sqrt(np.mean(resid ** 2)))
        mae   = float(np.mean(np.abs(resid)))
        return rmse, mae

    def naive(self, steps: int) -> ForecastResult:
        # Baseline: der letzte beobachtete Wert wird konstant fortgeschrieben.
        # In-Sample-Fit = Ein-Schritt-Prognose (jeder Punkt wird durch seinen Vorgänger geschätzt).
        fitted    = np.empty(self._n)
        fitted[0] = self._y[0]
        fitted[1:] = self._y[:-1]
        rmse, mae = self._fit_errors(self._y, fitted)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        return ForecastResult(
            method="Naive Prognose",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=np.full(steps, self._y[-1]), y_fitted=fitted,
            rmse=rmse, mae=mae,
        )

    def linear(self, steps: int) -> ForecastResult:
        slope, intercept, r, *_ = scipy_stats.linregress(self._x, self._y)
        fitted    = slope * self._x + intercept
        rmse, mae = self._fit_errors(self._y, fitted)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        y_fut  = slope * x_fut + intercept
        denom  = np.sum((self._x - self._x.mean()) ** 2)
        se     = rmse * np.sqrt(
            1 + 1 / self._n + (x_fut - self._x.mean()) ** 2 / denom
        )
        return ForecastResult(
            method="Linearer Trend",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=fitted,
            ci_low=y_fut - 1.96 * se, ci_high=y_fut + 1.96 * se,
            r2=r ** 2, rmse=rmse, mae=mae, slope=slope,
        )

    def moving_average(self, window: int, steps: int) -> ForecastResult:
        w         = min(window, self._n)
        rolling   = pd.Series(self._y).rolling(w, min_periods=1).mean().to_numpy()
        y_fut     = np.full(steps, np.mean(self._y[-w:]))
        x_fut     = np.arange(self._n, self._n + steps, dtype=float)
        rmse, mae = self._fit_errors(self._y, rolling)
        return ForecastResult(
            method=f"Gleitender Durchschnitt (Fenster={window})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=rolling,
            rmse=rmse, mae=mae,
        )

    def exp_smoothing(self, alpha: float, steps: int) -> ForecastResult:
        smoothed  = self._smooth(alpha)
        x_fut     = np.arange(self._n, self._n + steps, dtype=float)
        rmse, mae = self._fit_errors(self._y, smoothed)
        return ForecastResult(
            method=f"Exponentielle Glättung (α={alpha})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=np.full(steps, smoothed[-1]), y_fitted=smoothed,
            rmse=rmse, mae=mae,
        )

    def arima(self, p: int, d: int, q: int, steps: int) -> ForecastResult:
        model  = ARIMA(self._y, order=(p, d, q)).fit()
        fc     = model.get_forecast(steps=steps)
        y_fut  = fc.predicted_mean
        ci     = fc.conf_int(alpha=0.05)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        fitted    = model.fittedvalues
        rmse, mae = self._fit_errors(self._y, fitted)
        return ForecastResult(
            method=f"ARIMA({p},{d},{q})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=fitted,
            ci_low=ci[:, 0], ci_high=ci[:, 1],
            rmse=rmse, mae=mae,
        )

    def holt_winters(self, period: int, seasonal: str, steps: int) -> ForecastResult:
        model  = ExponentialSmoothing(
            self._y, trend="add", seasonal=seasonal, seasonal_periods=period,
        ).fit(optimized=True)
        y_fut  = model.forecast(steps)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        fitted    = model.fittedvalues
        rmse, mae = self._fit_errors(self._y, fitted)
        return ForecastResult(
            method=f"Holt-Winters (Periode={period}, {seasonal})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=fitted,
            rmse=rmse, mae=mae,
        )

    def _smooth(self, alpha: float) -> np.ndarray:
        s = [self._y[0]]
        for v in self._y[1:]:
            s.append(alpha * v + (1 - alpha) * s[-1])
        return np.array(s)
