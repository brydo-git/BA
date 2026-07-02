import matplotlib.pyplot as plt

from forecaster import ForecastResult


class ChartRenderer:
    @staticmethod
    def forecast(result: ForecastResult, col_name: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        ax.plot(result.x_num, result.y_hist, "o-", ms=3, label="Historisch", alpha=0.7)
        if result.y_fitted is not None:
            ax.plot(result.x_num, result.y_fitted, "--", label="Fit / Geglättet", lw=1.4)
        ax.plot(result.x_fut, result.y_fut, "r-", label="Prognose", lw=1.8)
        if result.ci_low is not None:
            ax.fill_between(
                result.x_fut, result.ci_low, result.ci_high,
                alpha=0.12, color="red", label="95%-KI",
            )
        ax.axvline(x=len(result.y_hist) - 0.5, color="gray",
                   linestyle=":", lw=1, label="Heute")

        title = f"{result.method} – {col_name}"
        if result.r2 is not None:
            title += f"  |  R²={result.r2:.3f}  |  RMSE={result.rmse:.2f}"
        ax.set_title(title)
        ax.legend()
        return fig
