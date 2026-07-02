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

    def linear(self, steps: int) -> ForecastResult:
        slope, intercept, r, *_ = scipy_stats.linregress(self._x, self._y)
        fitted = slope * self._x + intercept
        rmse   = np.sqrt(np.mean((self._y - fitted) ** 2))
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
            r2=r ** 2, rmse=rmse, slope=slope,
        )

    def moving_average(self, window: int, steps: int) -> ForecastResult:
        w       = min(window, self._n)
        rolling = pd.Series(self._y).rolling(w, min_periods=1).mean().to_numpy()
        y_fut   = np.full(steps, np.mean(self._y[-w:]))
        x_fut   = np.arange(self._n, self._n + steps, dtype=float)
        return ForecastResult(
            method=f"Gleitender Durchschnitt (Fenster={window})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=rolling,
        )

    def exp_smoothing(self, alpha: float, steps: int) -> ForecastResult:
        smoothed = self._smooth(alpha)
        x_fut    = np.arange(self._n, self._n + steps, dtype=float)
        return ForecastResult(
            method=f"Exponentielle Glättung (α={alpha})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=np.full(steps, smoothed[-1]), y_fitted=smoothed,
        )

    def arima(self, p: int, d: int, q: int, steps: int) -> ForecastResult:
        model  = ARIMA(self._y, order=(p, d, q)).fit()
        fc     = model.get_forecast(steps=steps)
        y_fut  = fc.predicted_mean
        ci     = fc.conf_int(alpha=0.05)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        fitted = model.fittedvalues
        resid  = self._y - fitted
        rmse   = np.sqrt(np.mean(resid ** 2))
        return ForecastResult(
            method=f"ARIMA({p},{d},{q})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=fitted,
            ci_low=ci[:, 0], ci_high=ci[:, 1],
            rmse=rmse,
        )

    def holt_winters(self, period: int, seasonal: str, steps: int) -> ForecastResult:
        model  = ExponentialSmoothing(
            self._y, trend="add", seasonal=seasonal, seasonal_periods=period,
        ).fit(optimized=True)
        y_fut  = model.forecast(steps)
        x_fut  = np.arange(self._n, self._n + steps, dtype=float)
        fitted = model.fittedvalues
        rmse   = np.sqrt(np.mean((self._y - fitted) ** 2))
        return ForecastResult(
            method=f"Holt-Winters (Periode={period}, {seasonal})",
            y_hist=self._y, x_num=self._x,
            x_fut=x_fut, y_fut=y_fut, y_fitted=fitted,
            rmse=rmse,
        )

    def _smooth(self, alpha: float) -> np.ndarray:
        s = [self._y[0]]
        for v in self._y[1:]:
            s.append(alpha * v + (1 - alpha) * s[-1])
        return np.array(s)
