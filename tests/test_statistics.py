import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from statistics import StatisticsAnalyzer


@pytest.fixture
def df():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


class TestCompute:
    def test_mean(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Mittelwert"])
        assert result.loc["x", "Mittelwert"] == pytest.approx(3.0)

    def test_median(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Median"])
        assert result.loc["x", "Median"] == pytest.approx(3.0)

    def test_std(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Standardabweichung"])
        expected = np.std([1, 2, 3, 4, 5], ddof=1)
        assert result.loc["x", "Standardabweichung"] == pytest.approx(expected)

    def test_min(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Minimum"])
        assert result.loc["x", "Minimum"] == 1.0

    def test_max(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Maximum"])
        assert result.loc["x", "Maximum"] == 5.0

    def test_q25(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["25%-Quantil"])
        assert result.loc["x", "25%-Quantil"] == pytest.approx(2.0)

    def test_q75(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["75%-Quantil"])
        assert result.loc["x", "75%-Quantil"] == pytest.approx(4.0)

    def test_count(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Anzahl (n)"])
        assert result.loc["x", "Anzahl (n)"] == 5

    def test_multiple_metrics(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute(["Mittelwert", "Minimum"])
        assert "Mittelwert" in result.columns
        assert "Minimum" in result.columns

    def test_multiple_columns(self, df):
        result = StatisticsAnalyzer(df, ["x", "y"]).compute(["Mittelwert"])
        assert result.loc["x", "Mittelwert"] == pytest.approx(3.0)
        assert result.loc["y", "Mittelwert"] == pytest.approx(30.0)

    def test_non_numeric_values_coerced(self):
        df = pd.DataFrame({"x": ["1", "2", "abc", "4"]})
        result = StatisticsAnalyzer(df, ["x"]).compute(["Anzahl (n)"])
        assert result.loc["x", "Anzahl (n)"] == 3

    def test_empty_metrics_returns_empty_df(self, df):
        result = StatisticsAnalyzer(df, ["x"]).compute([])
        assert result.empty


class TestInvalidCounts:
    def test_no_invalid_values(self, df):
        assert StatisticsAnalyzer(df, ["x", "y"]).invalid_counts() == {"x": 0, "y": 0}

    def test_counts_non_numeric_and_missing(self):
        df = pd.DataFrame({"x": ["1", "abc", None, "4"]})
        assert StatisticsAnalyzer(df, ["x"]).invalid_counts() == {"x": 2}

    def test_counts_per_column_independently(self):
        df = pd.DataFrame({"x": ["1", "abc"], "y": ["10", "20"]})
        assert StatisticsAnalyzer(df, ["x", "y"]).invalid_counts() == {"x": 1, "y": 0}


class TestBarCharts:
    def test_yields_one_figure_per_metric(self, df):
        analyzer = StatisticsAnalyzer(df, ["x"])
        stats = analyzer.compute(["Mittelwert", "Median"])
        items = list(analyzer.bar_charts(stats))
        assert len(items) == 2

    def test_metric_names_match(self, df):
        analyzer = StatisticsAnalyzer(df, ["x"])
        stats = analyzer.compute(["Mittelwert", "Maximum"])
        names = [m for m, _ in analyzer.bar_charts(stats)]
        assert names == ["Mittelwert", "Maximum"]

    def test_yields_matplotlib_figures(self, df):
        analyzer = StatisticsAnalyzer(df, ["x"])
        stats = analyzer.compute(["Mittelwert"])
        for _, fig in analyzer.bar_charts(stats):
            assert isinstance(fig, plt.Figure)


class TestBoxplots:
    def test_yields_one_figure_per_column(self, df):
        analyzer = StatisticsAnalyzer(df, ["x", "y"])
        items = list(analyzer.boxplots())
        assert len(items) == 2

    def test_column_names_match(self, df):
        analyzer = StatisticsAnalyzer(df, ["x", "y"])
        names = [col for col, _ in analyzer.boxplots()]
        assert names == ["x", "y"]

    def test_yields_matplotlib_figures(self, df):
        analyzer = StatisticsAnalyzer(df, ["x"])
        for _, fig in analyzer.boxplots():
            assert isinstance(fig, plt.Figure)


class TestGroupedStats:
    @staticmethod
    def _data():
        return pd.DataFrame({
            "Werk":  ["A", "A", "A", "A", "B", "B"],
            "Menge": [1, 2, 3, 10, 5, 7],
        })

    def _run(self, agg):
        df = self._data()
        return StatisticsAnalyzer(df, ["Menge"]).grouped_stats(["Werk"], df, agg)["Menge"]

    def test_median(self):
        r = self._run("Median")
        assert r["A"] == 2.5 and r["B"] == 6.0

    def test_std_matches_statistics_section(self):
        r = self._run("Standardabweichung")
        assert r["A"] == pytest.approx(pd.Series([1, 2, 3, 10]).std())
        assert r["B"] == pytest.approx(pd.Series([5, 7]).std())

    def test_quantiles(self):
        assert self._run("25%-Quantil")["A"] == pytest.approx(1.75)
        assert self._run("75%-Quantil")["A"] == pytest.approx(4.75)
        assert self._run("25%-Quantil")["B"] == pytest.approx(5.5)

    def test_std_single_value_group_is_empty(self):
        df = pd.DataFrame({"Werk": ["A", "A", "B"], "Menge": [1, 3, 9]})
        r = StatisticsAnalyzer(df, ["Menge"]).grouped_stats(["Werk"], df, "Standardabweichung")["Menge"]
        assert pd.isna(r["B"])

    def test_multiple_group_and_value_columns(self):
        df = pd.DataFrame({"Werk": ["A", "A", "B", "B"], "BwA": ["101", "101", "101", "601"],
                           "Menge": [1, 3, 5, -7], "Wert": [10, 30, 50, 70]})
        r = StatisticsAnalyzer(df, ["Menge", "Wert"]).grouped_stats(["Werk", "BwA"], df, "75%-Quantil")
        assert list(r.columns) == ["Menge", "Wert"]
        assert r.loc[("A", "101"), "Menge"] == pytest.approx(2.5)

    def test_all_aggregations_run(self):
        df = self._data()
        a = StatisticsAnalyzer(df, ["Menge"])
        for agg in StatisticsAnalyzer.AGG_NAMES:
            assert len(a.grouped_stats(["Werk"], df, agg)) == 2
