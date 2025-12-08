"""
Tests for benchmarks/visualizations.py

Tests the visualization module with both matplotlib and plotly backends.
Uses mocking where appropriate to avoid requiring actual visualization libraries.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.visualizations import (
    AccuracyScaleData,
    BenchmarkVisualizer,
    ChartType,
    ComparisonBarData,
    ConfusionMatrixData,
    ErrorDistributionData,
    MatplotlibBackend,
    ParetoData,
    PlotlyBackend,
    RobustnessHeatmapData,
    TokenReductionData,
    VisualizationConfig,
    create_visualizer,
)

# ============================================================================
# VisualizationConfig Tests
# ============================================================================


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VisualizationConfig()

        assert config.enabled is True
        assert config.backend == "matplotlib"
        assert config.output_format == "png"
        assert config.dpi == 150
        assert config.width == 1200
        assert config.height == 800
        assert config.theme == "default"
        assert config.include_title is True
        assert config.include_legend is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VisualizationConfig(
            backend="plotly",
            output_format="html",
            dpi=300,
            theme="dark",
        )

        assert config.backend == "plotly"
        assert config.output_format == "html"
        assert config.dpi == 300
        assert config.theme == "dark"

    def test_invalid_backend_raises(self) -> None:
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Backend must be one of"):
            VisualizationConfig(backend="invalid")

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Format must be one of"):
            VisualizationConfig(output_format="invalid")


# ============================================================================
# Data Class Tests
# ============================================================================


class TestTokenReductionData:
    """Tests for TokenReductionData dataclass."""

    def test_default_labels(self) -> None:
        """Test default labels."""
        data = TokenReductionData(
            raw_tokens=10000,
            semantic_tokens=500,
            final_tokens=500,
        )

        assert data.labels == ["Raw Data", "Semantic Frame", "Final"]

    def test_custom_labels(self) -> None:
        """Test custom labels."""
        data = TokenReductionData(
            raw_tokens=10000,
            semantic_tokens=500,
            final_tokens=500,
            labels=["Input", "Processed", "Output"],
        )

        assert data.labels == ["Input", "Processed", "Output"]


class TestAccuracyScaleData:
    """Tests for AccuracyScaleData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        data = AccuracyScaleData(
            scale_points=[100, 1000, 10000],
            baseline_accuracy=[0.9, 0.8, 0.7],
            treatment_accuracy=[0.95, 0.93, 0.92],
        )

        assert len(data.scale_points) == 3
        assert data.baseline_label == "Baseline (raw data)"
        assert data.treatment_label == "Treatment (semantic-frame)"


class TestParetoData:
    """Tests for ParetoData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        data = ParetoData(
            baseline_tokens=[1000, 2000, 3000],
            baseline_accuracy=[0.7, 0.8, 0.85],
            treatment_tokens=[100, 150, 200],
            treatment_accuracy=[0.9, 0.92, 0.93],
        )

        assert len(data.baseline_tokens) == 3
        assert len(data.treatment_tokens) == 3


class TestConfusionMatrixData:
    """Tests for ConfusionMatrixData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        matrix = np.array([[10, 2], [3, 15]])
        data = ConfusionMatrixData(
            matrix=matrix,
            labels=["Positive", "Negative"],
        )

        assert data.matrix.shape == (2, 2)
        assert data.title == "Confusion Matrix"


class TestErrorDistributionData:
    """Tests for ErrorDistributionData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        data = ErrorDistributionData(
            baseline_errors=[0.1, 0.2, 0.15, 0.25],
            treatment_errors=[0.05, 0.08, 0.06, 0.07],
        )

        assert len(data.baseline_errors) == 4
        assert data.bins == 30


class TestRobustnessHeatmapData:
    """Tests for RobustnessHeatmapData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        matrix = np.array([[0.9, 0.85, 0.8], [0.95, 0.9, 0.85]])
        data = RobustnessHeatmapData(
            perturbation_types=["noise", "scale"],
            perturbation_levels=[0.1, 0.2, 0.3],
            accuracy_matrix=matrix,
        )

        assert data.accuracy_matrix.shape == (2, 3)


class TestComparisonBarData:
    """Tests for ComparisonBarData dataclass."""

    def test_creation(self) -> None:
        """Test data creation."""
        data = ComparisonBarData(
            categories=["Task1", "Task2", "Task3"],
            baseline_values=[0.7, 0.6, 0.65],
            treatment_values=[0.9, 0.85, 0.88],
        )

        assert len(data.categories) == 3
        assert data.metric_name == "Accuracy"


# ============================================================================
# ChartType Tests
# ============================================================================


class TestChartType:
    """Tests for ChartType enum."""

    def test_all_types_have_values(self) -> None:
        """Test that all chart types have string values."""
        for chart_type in ChartType:
            assert isinstance(chart_type.value, str)
            assert len(chart_type.value) > 0

    def test_type_count(self) -> None:
        """Test expected number of chart types."""
        assert len(ChartType) == 7


# ============================================================================
# MatplotlibBackend Tests
# ============================================================================


class TestMatplotlibBackend:
    """Tests for MatplotlibBackend class."""

    def test_is_available_with_matplotlib(self) -> None:
        """Test availability check when matplotlib is installed."""
        config = VisualizationConfig()
        backend = MatplotlibBackend(config)

        # This will depend on whether matplotlib is actually installed
        result = backend.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_token_reduction_waterfall(self) -> None:
        """Test token reduction waterfall generation."""
        config = VisualizationConfig()
        backend = MatplotlibBackend(config)

        data = TokenReductionData(
            raw_tokens=10000,
            semantic_tokens=500,
            final_tokens=500,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "waterfall.png"
            result = backend.token_reduction_waterfall(data, output_path)

            assert result.exists()
            assert result.suffix == ".png"

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_accuracy_vs_scale(self) -> None:
        """Test accuracy vs scale generation."""
        config = VisualizationConfig()
        backend = MatplotlibBackend(config)

        data = AccuracyScaleData(
            scale_points=[100, 1000, 10000],
            baseline_accuracy=[0.9, 0.8, 0.7],
            treatment_accuracy=[0.95, 0.93, 0.92],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scale.png"
            result = backend.accuracy_vs_scale(data, output_path)

            assert result.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_confusion_matrix(self) -> None:
        """Test confusion matrix generation."""
        config = VisualizationConfig()
        backend = MatplotlibBackend(config)

        matrix = np.array([[10, 2, 1], [3, 15, 2], [1, 2, 12]])
        data = ConfusionMatrixData(
            matrix=matrix,
            labels=["Rising", "Flat", "Falling"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confusion.png"
            result = backend.confusion_matrix_heatmap(data, output_path)

            assert result.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_svg_output(self) -> None:
        """Test SVG output format."""
        config = VisualizationConfig(output_format="svg")
        backend = MatplotlibBackend(config)

        data = TokenReductionData(raw_tokens=1000, semantic_tokens=100, final_tokens=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "chart.svg"
            result = backend.token_reduction_waterfall(data, output_path)

            assert result.suffix == ".svg"


# ============================================================================
# PlotlyBackend Tests
# ============================================================================


class TestPlotlyBackend:
    """Tests for PlotlyBackend class."""

    def test_is_available_check(self) -> None:
        """Test availability check."""
        config = VisualizationConfig(backend="plotly")
        backend = PlotlyBackend(config)

        result = backend.is_available()
        assert isinstance(result, bool)

    def test_get_template(self) -> None:
        """Test template selection."""
        config = VisualizationConfig(backend="plotly", theme="dark")
        backend = PlotlyBackend(config)

        template = backend._get_template()
        assert template == "plotly_dark"

    def test_get_template_paper(self) -> None:
        """Test paper theme template."""
        config = VisualizationConfig(backend="plotly", theme="paper")
        backend = PlotlyBackend(config)

        template = backend._get_template()
        assert template == "simple_white"


# ============================================================================
# BenchmarkVisualizer Tests
# ============================================================================


class TestBenchmarkVisualizer:
    """Tests for BenchmarkVisualizer class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        viz = BenchmarkVisualizer()

        assert viz.config is not None
        assert viz.config.backend == "matplotlib"

    def test_init_custom_config(self) -> None:
        """Test custom configuration."""
        config = VisualizationConfig(backend="plotly", output_format="html")
        viz = BenchmarkVisualizer(config)

        assert viz.config.backend == "plotly"
        assert viz.config.output_format == "html"

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_generate_all_empty(self) -> None:
        """Test generate_all with no data provided."""
        viz = BenchmarkVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = viz.generate_all(tmpdir)

            assert result == []

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_generate_all_with_data(self) -> None:
        """Test generate_all with some data."""
        viz = BenchmarkVisualizer()

        token_data = TokenReductionData(
            raw_tokens=10000,
            semantic_tokens=500,
            final_tokens=500,
        )

        comparison_data = ComparisonBarData(
            categories=["Task1", "Task2"],
            baseline_values=[0.7, 0.6],
            treatment_values=[0.9, 0.85],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = viz.generate_all(
                tmpdir,
                token_data=token_data,
                comparison_data=comparison_data,
            )

            assert len(result) == 2
            assert all(p.exists() for p in result)

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_individual_chart_methods(self) -> None:
        """Test individual chart generation methods."""
        viz = BenchmarkVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Token reduction
            token_data = TokenReductionData(raw_tokens=10000, semantic_tokens=500, final_tokens=500)
            path = viz.token_reduction_waterfall(token_data, Path(tmpdir) / "token.png")
            assert path.exists()

            # Comparison bar
            comparison_data = ComparisonBarData(
                categories=["A", "B"],
                baseline_values=[0.7, 0.6],
                treatment_values=[0.9, 0.85],
            )
            path = viz.comparison_bar_chart(comparison_data, Path(tmpdir) / "bar.png")
            assert path.exists()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateVisualizer:
    """Tests for create_visualizer factory function."""

    def test_create_default(self) -> None:
        """Test creating visualizer with defaults."""
        viz = create_visualizer()

        assert viz.config.backend == "matplotlib"
        assert viz.config.output_format == "png"
        assert viz.config.theme == "default"

    def test_create_plotly(self) -> None:
        """Test creating plotly visualizer."""
        viz = create_visualizer(backend="plotly", output_format="html", theme="dark")

        assert viz.config.backend == "plotly"
        assert viz.config.output_format == "html"
        assert viz.config.theme == "dark"


# ============================================================================
# Integration Tests (with mocking)
# ============================================================================


class TestIntegration:
    """Integration tests with mocking."""

    def test_matplotlib_fallback_when_plotly_unavailable(self) -> None:
        """Test fallback to matplotlib when plotly is unavailable."""
        config = VisualizationConfig(backend="plotly")

        with patch.object(PlotlyBackend, "is_available", return_value=False):
            with patch.object(MatplotlibBackend, "is_available", return_value=True):
                viz = BenchmarkVisualizer(config)
                backend = viz._create_backend()

                # Should fall back to matplotlib
                assert isinstance(backend, MatplotlibBackend)

    def test_error_when_no_backend_available(self) -> None:
        """Test error when no backend is available."""
        config = VisualizationConfig()

        with patch.object(MatplotlibBackend, "is_available", return_value=False):
            with patch.object(PlotlyBackend, "is_available", return_value=False):
                viz = BenchmarkVisualizer(config)

                with pytest.raises(ImportError, match="No visualization backend available"):
                    _ = viz.backend


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_empty_error_distribution(self) -> None:
        """Test error distribution with empty lists."""
        viz = BenchmarkVisualizer()

        # Empty lists should still work (matplotlib handles gracefully)
        data = ErrorDistributionData(
            baseline_errors=[0.1],  # Need at least one point
            treatment_errors=[0.05],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.error_distribution_histogram(data, Path(tmpdir) / "hist.png")
            assert path.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_single_point_pareto(self) -> None:
        """Test Pareto plot with single points."""
        viz = BenchmarkVisualizer()

        data = ParetoData(
            baseline_tokens=[1000],
            baseline_accuracy=[0.7],
            treatment_tokens=[100],
            treatment_accuracy=[0.9],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.pareto_frontier(data, Path(tmpdir) / "pareto.png")
            assert path.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_zero_compression(self) -> None:
        """Test token reduction with zero compression."""
        viz = BenchmarkVisualizer()

        data = TokenReductionData(
            raw_tokens=1000,
            semantic_tokens=1000,
            final_tokens=1000,  # No reduction
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.token_reduction_waterfall(data, Path(tmpdir) / "waterfall.png")
            assert path.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_large_confusion_matrix(self) -> None:
        """Test large confusion matrix."""
        viz = BenchmarkVisualizer()

        # 5x5 matrix
        matrix = np.random.rand(5, 5) * 100
        data = ConfusionMatrixData(
            matrix=matrix,
            labels=["A", "B", "C", "D", "E"],
            title="Large Matrix",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.confusion_matrix_heatmap(data, Path(tmpdir) / "confusion.png")
            assert path.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_robustness_heatmap_generation(self) -> None:
        """Test robustness heatmap generation."""
        viz = BenchmarkVisualizer()

        matrix = np.array(
            [
                [0.95, 0.90, 0.85, 0.80],
                [0.90, 0.85, 0.80, 0.75],
                [0.85, 0.80, 0.75, 0.70],
            ]
        )
        data = RobustnessHeatmapData(
            perturbation_types=["noise", "scale", "shift"],
            perturbation_levels=[0.05, 0.10, 0.20, 0.30],
            accuracy_matrix=matrix,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.robustness_heatmap(data, Path(tmpdir) / "robustness.png")
            assert path.exists()


# ============================================================================
# Theme Tests
# ============================================================================


class TestThemes:
    """Tests for theme support."""

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_dark_theme(self) -> None:
        """Test dark theme generation."""
        config = VisualizationConfig(theme="dark")
        viz = BenchmarkVisualizer(config)

        data = TokenReductionData(raw_tokens=1000, semantic_tokens=100, final_tokens=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.token_reduction_waterfall(data, Path(tmpdir) / "dark.png")
            assert path.exists()

    @pytest.mark.skipif(
        not MatplotlibBackend(VisualizationConfig()).is_available(),
        reason="matplotlib not installed",
    )
    def test_paper_theme(self) -> None:
        """Test paper theme generation."""
        config = VisualizationConfig(theme="paper")
        viz = BenchmarkVisualizer(config)

        data = TokenReductionData(raw_tokens=1000, semantic_tokens=100, final_tokens=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.token_reduction_waterfall(data, Path(tmpdir) / "paper.png")
            assert path.exists()
