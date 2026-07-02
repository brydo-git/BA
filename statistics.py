import pandas as pd
import matplotlib.pyplot as plt


class StatisticsAnalyzer:
    METRICS: dict = {
        "Mittelwert":         lambda d: d.mean(),
        "Median":             lambda d: d.median(),
        "Standardabweichung": lambda d: d.std(),
        "Minimum":            lambda d: d.min(),
        "Maximum":            lambda d: d.max(),
        "25%-Quantil":        lambda d: d.quantile(0.25),
        "75%-Quantil":        lambda d: d.quantile(0.75),
        "Anzahl (n)":         lambda d: d.count(),
    }

    METRIC_NAMES = list(METRICS.keys())

    def __init__(self, df: pd.DataFrame, columns: list):
        self._data = df[columns].apply(pd.to_numeric, errors="coerce")

    def invalid_counts(self) -> dict:
        """Count of missing/non-numeric values per column, excluded from all calculations."""
        return {c: int(self._data[c].isna().sum()) for c in self._data.columns}

    def compute(self, metrics: list) -> pd.DataFrame:
        return pd.DataFrame(
            {m: self.METRICS[m](self._data) for m in metrics}
        )

    def bar_charts(self, stats_df: pd.DataFrame):
        for metric in stats_df.columns:
            fig, ax = plt.subplots()
            bars = ax.bar(stats_df.index.astype(str), stats_df[metric], color="#4C8BF5")
            ax.bar_label(bars, fmt=lambda v: f"{v:,.2f}", padding=3, fontsize=8)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=30)
            ax.margins(y=0.12)
            yield metric, fig
            plt.close(fig)

    def boxplots(self):
        for col in self._data.columns:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.boxplot(
                self._data[col].dropna(), orientation="vertical", patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"),
            )
            ax.set_title(col)
            fig.tight_layout()
            yield col, fig
            plt.close(fig)

    # ── Gruppenstatistik ──────────────────────────────────────────────────────

    AGG_MAP = {
        "Summe":      "sum",
        "Mittelwert": "mean",
        "Median":     "median",
        "Anzahl":     "count",
        "Minimum":    "min",
        "Maximum":    "max",
    }
    AGG_NAMES = list(AGG_MAP.keys())

    def grouped_stats(self, group_cols: list, source_df: pd.DataFrame, agg: str) -> pd.DataFrame:
        """Aggregate numeric columns grouped by one or more categorical columns."""
        tmp = source_df[group_cols].reset_index(drop=True).copy()
        for col in self._data.columns:
            tmp[col] = self._data[col].values
        result = tmp.groupby(group_cols)[self._data.columns.tolist()].agg(self.AGG_MAP[agg])
        return result.sort_values(self._data.columns[0], ascending=False)

    def grouped_bar_chart(self, result: pd.DataFrame, agg: str) -> plt.Figure:
        cols = result.columns.tolist()
        if isinstance(result.index, pd.MultiIndex):
            x_labels = [" | ".join(map(str, t)) for t in result.index]
            group_name = " & ".join(result.index.names)
        else:
            x_labels = result.index.astype(str).tolist()
            group_name = result.index.name

        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(result))
        width = 0.8 / max(len(cols), 1)
        colors = ["#4C8BF5", "#F5A623", "#27AE60", "#E74C3C", "#9B59B6"]
        for i, col in enumerate(cols):
            offset = (i - len(cols) / 2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], result[col], width=width,
                          label=col, color=colors[i % len(colors)])
            ax.bar_label(bars, fmt=lambda v: f"{v:,.2f}", padding=3, fontsize=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_title(f"{agg} nach {group_name}")
        ax.margins(y=0.15)
        if len(cols) > 1:
            ax.legend()
        fig.tight_layout()
        return fig
