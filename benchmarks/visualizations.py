"""
Benchmark Visualizations

Dual-backend visualization module supporting both matplotlib (static)
and plotly (interactive) chart generation for benchmark results.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of benchmark visualizations."""

    TOKEN_REDUCTION_WATERFALL = "token_reduction_waterfall"
    ACCURACY_VS_SCALE = "accuracy_vs_scale"
    PARETO_FRONTIER = "pareto_frontier"
    CONFUSION_MATRIX = "confusion_matrix"
    ERROR_DISTRIBUTION = "error_distribution"
    ROBUSTNESS_HEATMAP = "robustness_heatmap"
    COMPARISON_BAR = "comparison_bar"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""

    enabled: bool = True
    backend: str = "matplotlib"  # "matplotlib" or "plotly"
    output_format: str = "png"  # "png", "svg", "html", "pdf"
    dpi: int = 150
    width: int = 1200
    height: int = 800
    theme: str = "default"  # "default", "dark", "paper"
    include_title: bool = True
    include_legend: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_backends = {"matplotlib", "plotly"}
        if self.backend not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}, got {self.backend}")

        valid_formats = {"png", "svg", "html", "pdf"}
        if self.output_format not in valid_formats:
            raise ValueError(f"Format must be one of {valid_formats}, got {self.output_format}")


@dataclass
class TokenReductionData:
    """Data for token reduction waterfall chart."""

    raw_tokens: int
    semantic_tokens: int
    final_tokens: int
    labels: list[str] = field(default_factory=lambda: ["Raw Data", "Semantic Frame", "Final"])


@dataclass
class AccuracyScaleData:
    """Data for accuracy vs scale plot."""

    scale_points: list[int]  # Data sizes
    baseline_accuracy: list[float]
    treatment_accuracy: list[float]
    baseline_label: str = "Baseline (raw data)"
    treatment_label: str = "Treatment (semantic-frame)"


@dataclass
class ParetoData:
    """Data for Pareto frontier plot."""

    baseline_tokens: list[int]
    baseline_accuracy: list[float]
    treatment_tokens: list[int]
    treatment_accuracy: list[float]


@dataclass
class ConfusionMatrixData:
    """Data for confusion matrix heatmap."""

    matrix: NDArray[np.float64]  # NxN matrix of values
    labels: list[str]  # Class labels
    title: str = "Confusion Matrix"


@dataclass
class ErrorDistributionData:
    """Data for error distribution histogram."""

    baseline_errors: list[float]
    treatment_errors: list[float]
    bins: int = 30


@dataclass
class RobustnessHeatmapData:
    """Data for robustness heatmap."""

    perturbation_types: list[str]
    perturbation_levels: list[float]
    accuracy_matrix: NDArray[np.float64]  # types x levels


@dataclass
class ComparisonBarData:
    """Data for comparison bar chart."""

    categories: list[str]
    baseline_values: list[float]
    treatment_values: list[float]
    metric_name: str = "Accuracy"


class VisualizationBackend(ABC):
    """Abstract backend for chart generation."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    @abstractmethod
    def token_reduction_waterfall(self, data: TokenReductionData, output_path: Path) -> Path:
        """Generate token reduction waterfall chart."""
        pass

    @abstractmethod
    def accuracy_vs_scale(self, data: AccuracyScaleData, output_path: Path) -> Path:
        """Generate accuracy vs scale line plot."""
        pass

    @abstractmethod
    def pareto_frontier(self, data: ParetoData, output_path: Path) -> Path:
        """Generate Pareto frontier scatter plot."""
        pass

    @abstractmethod
    def confusion_matrix_heatmap(self, data: ConfusionMatrixData, output_path: Path) -> Path:
        """Generate confusion matrix heatmap."""
        pass

    @abstractmethod
    def error_distribution_histogram(self, data: ErrorDistributionData, output_path: Path) -> Path:
        """Generate error distribution histogram."""
        pass

    @abstractmethod
    def robustness_heatmap(self, data: RobustnessHeatmapData, output_path: Path) -> Path:
        """Generate robustness heatmap."""
        pass

    @abstractmethod
    def comparison_bar_chart(self, data: ComparisonBarData, output_path: Path) -> Path:
        """Generate comparison bar chart."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if required dependencies are installed."""
        pass


class MatplotlibBackend(VisualizationBackend):
    """
    Matplotlib-based backend for static PNG/SVG/PDF output.

    Best for: Reports, papers, static documentation.
    """

    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        self._plt: Any = None
        self._np = np

    def is_available(self) -> bool:
        """Check if matplotlib is installed."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_plt(self) -> Any:
        """Lazy import matplotlib."""
        if self._plt is None:
            try:
                import matplotlib.pyplot as plt

                self._plt = plt

                # Apply theme
                if self.config.theme == "dark":
                    plt.style.use("dark_background")
                elif self.config.theme == "paper":
                    plt.style.use("seaborn-v0_8-paper")
                else:
                    plt.style.use("seaborn-v0_8-whitegrid")

            except ImportError as e:
                raise ImportError(
                    "matplotlib is required for visualization. "
                    "Install with: pip install semantic-frame[viz]"
                ) from e
        return self._plt

    def _save_figure(self, fig: Any, output_path: Path) -> Path:
        """Save figure and clean up."""
        plt = self._get_plt()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Adjust extension based on format
        if self.config.output_format != output_path.suffix.lstrip("."):
            output_path = output_path.with_suffix(f".{self.config.output_format}")

        fig.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches="tight",
            facecolor="white" if self.config.theme != "dark" else "black",
        )
        plt.close(fig)
        return output_path

    def token_reduction_waterfall(self, data: TokenReductionData, output_path: Path) -> Path:
        """Generate token reduction waterfall chart."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate values for each bar
        cumulative = [data.raw_tokens, data.semantic_tokens, data.final_tokens]

        colors = ["#4CAF50", "#F44336", "#2196F3"]

        # Create waterfall effect
        x = range(len(data.labels))
        bars = ax.bar(x, [data.raw_tokens, data.semantic_tokens, data.final_tokens], color=colors)

        # Add value labels on bars
        for bar, val in zip(bars, cumulative, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Add compression ratio annotation
        if data.raw_tokens > 0:
            ratio = (1 - data.final_tokens / data.raw_tokens) * 100
            ax.text(
                0.95,
                0.95,
                f"Compression: {ratio:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox={"boxstyle": "round", "facecolor": "wheat"},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data.labels)
        ax.set_ylabel("Tokens")
        if self.config.include_title:
            ax.set_title("Token Reduction Waterfall")
        ax.set_ylim(0, data.raw_tokens * 1.1)

        return self._save_figure(fig, output_path)

    def accuracy_vs_scale(self, data: AccuracyScaleData, output_path: Path) -> Path:
        """Generate accuracy vs scale line plot."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            data.scale_points,
            data.baseline_accuracy,
            "o-",
            label=data.baseline_label,
            color="#F44336",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            data.scale_points,
            data.treatment_accuracy,
            "s-",
            label=data.treatment_label,
            color="#4CAF50",
            linewidth=2,
            markersize=8,
        )

        ax.set_xlabel("Data Size")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

        if self.config.include_legend:
            ax.legend(loc="lower left")
        if self.config.include_title:
            ax.set_title("Accuracy vs Data Scale")

        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, output_path)

    def pareto_frontier(self, data: ParetoData, output_path: Path) -> Path:
        """Generate Pareto frontier scatter plot."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plots
        ax.scatter(
            data.baseline_tokens,
            data.baseline_accuracy,
            c="#F44336",
            label="Baseline",
            alpha=0.7,
            s=100,
        )
        ax.scatter(
            data.treatment_tokens,
            data.treatment_accuracy,
            c="#4CAF50",
            label="Treatment",
            alpha=0.7,
            s=100,
            marker="s",
        )

        # Draw Pareto frontiers
        self._draw_pareto_frontier(ax, data.baseline_tokens, data.baseline_accuracy, "#F44336")
        self._draw_pareto_frontier(ax, data.treatment_tokens, data.treatment_accuracy, "#4CAF50")

        ax.set_xlabel("Tokens")
        ax.set_ylabel("Accuracy")
        ax.set_xscale("log")

        if self.config.include_legend:
            ax.legend()
        if self.config.include_title:
            ax.set_title("Pareto Frontier: Tokens vs Accuracy")

        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, output_path)

    def _draw_pareto_frontier(
        self,
        ax: Any,
        x_vals: list[int],
        y_vals: list[float],
        color: str,
    ) -> None:
        """Draw Pareto frontier line."""
        # Sort by x (tokens)
        sorted_pairs = sorted(zip(x_vals, y_vals, strict=False))
        pareto_x, pareto_y = [], []

        max_y = -float("inf")
        for x, y in sorted_pairs:
            if y > max_y:
                pareto_x.append(x)
                pareto_y.append(y)
                max_y = y

        if pareto_x:
            ax.plot(pareto_x, pareto_y, "--", color=color, alpha=0.5, linewidth=2)

    def confusion_matrix_heatmap(self, data: ConfusionMatrixData, output_path: Path) -> Path:
        """Generate confusion matrix heatmap."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(data.matrix, cmap="Blues")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(range(len(data.labels)))
        ax.set_yticks(range(len(data.labels)))
        ax.set_xticklabels(data.labels)
        ax.set_yticklabels(data.labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(data.labels)):
            for j in range(len(data.labels)):
                val = data.matrix[i, j]
                color = "white" if val > data.matrix.max() / 2 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if self.config.include_title:
            ax.set_title(data.title)

        return self._save_figure(fig, output_path)

    def error_distribution_histogram(self, data: ErrorDistributionData, output_path: Path) -> Path:
        """Generate error distribution histogram."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            data.baseline_errors,
            bins=data.bins,
            alpha=0.6,
            label="Baseline",
            color="#F44336",
        )
        ax.hist(
            data.treatment_errors,
            bins=data.bins,
            alpha=0.6,
            label="Treatment",
            color="#4CAF50",
        )

        # Add mean lines
        baseline_mean = np.mean(data.baseline_errors)
        treatment_mean = np.mean(data.treatment_errors)
        ax.axvline(baseline_mean, color="#F44336", linestyle="--", linewidth=2)
        ax.axvline(treatment_mean, color="#4CAF50", linestyle="--", linewidth=2)

        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")

        if self.config.include_legend:
            ax.legend()
        if self.config.include_title:
            ax.set_title("Error Distribution: Baseline vs Treatment")

        return self._save_figure(fig, output_path)

    def robustness_heatmap(self, data: RobustnessHeatmapData, output_path: Path) -> Path:
        """Generate robustness heatmap."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(data.accuracy_matrix, cmap="RdYlGn", vmin=0, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(range(len(data.perturbation_levels)))
        ax.set_yticks(range(len(data.perturbation_types)))
        ax.set_xticklabels([f"{level:.2f}" for level in data.perturbation_levels])
        ax.set_yticklabels(data.perturbation_types)

        # Add text annotations
        for i in range(len(data.perturbation_types)):
            for j in range(len(data.perturbation_levels)):
                val = data.accuracy_matrix[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

        ax.set_xlabel("Perturbation Level")
        ax.set_ylabel("Perturbation Type")
        if self.config.include_title:
            ax.set_title("Robustness Heatmap")

        return self._save_figure(fig, output_path)

    def comparison_bar_chart(self, data: ComparisonBarData, output_path: Path) -> Path:
        """Generate comparison bar chart."""
        plt = self._get_plt()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(data.categories))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            data.baseline_values,
            width,
            label="Baseline",
            color="#F44336",
        )
        bars2 = ax.bar(
            x + width / 2,
            data.treatment_values,
            width,
            label="Treatment",
            color="#4CAF50",
        )

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data.categories)
        ax.set_ylabel(data.metric_name)

        if self.config.include_legend:
            ax.legend()
        if self.config.include_title:
            ax.set_title(f"{data.metric_name} Comparison")

        return self._save_figure(fig, output_path)


class PlotlyBackend(VisualizationBackend):
    """
    Plotly-based backend for interactive HTML output.

    Best for: Dashboards, exploration, web presentation.
    """

    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        self._go: Any = None
        self._px: Any = None

    def is_available(self) -> bool:
        """Check if plotly is installed."""
        try:
            import plotly.graph_objects  # type: ignore[import-untyped]  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_plotly(self) -> tuple[Any, Any]:
        """Lazy import plotly."""
        if self._go is None:
            try:
                import plotly.express as px  # type: ignore[import-untyped]
                import plotly.graph_objects as go  # type: ignore[import-untyped]

                self._go = go
                self._px = px
            except ImportError as e:
                raise ImportError(
                    "plotly is required for interactive visualization. "
                    "Install with: pip install semantic-frame[viz]"
                ) from e
        return self._go, self._px

    def _get_template(self) -> str:
        """Get plotly template based on theme."""
        templates = {
            "default": "plotly_white",
            "dark": "plotly_dark",
            "paper": "simple_white",
        }
        return templates.get(self.config.theme, "plotly_white")

    def _save_figure(self, fig: Any, output_path: Path) -> Path:
        """Save figure to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Plotly always outputs HTML unless we use other formats
        if self.config.output_format == "html":
            output_path = output_path.with_suffix(".html")
            fig.write_html(output_path)
        elif self.config.output_format == "png":
            output_path = output_path.with_suffix(".png")
            fig.write_image(output_path, scale=2)
        elif self.config.output_format == "svg":
            output_path = output_path.with_suffix(".svg")
            fig.write_image(output_path)
        elif self.config.output_format == "pdf":
            output_path = output_path.with_suffix(".pdf")
            fig.write_image(output_path)
        else:
            output_path = output_path.with_suffix(".html")
            fig.write_html(output_path)

        return output_path

    def token_reduction_waterfall(self, data: TokenReductionData, output_path: Path) -> Path:
        """Generate interactive token reduction waterfall chart."""
        go, px = self._get_plotly()

        values = [data.raw_tokens, data.semantic_tokens, data.final_tokens]

        fig = go.Figure(
            go.Bar(
                x=data.labels,
                y=values,
                marker_color=["#4CAF50", "#FF9800", "#2196F3"],
                text=[f"{v:,}" for v in values],
                textposition="outside",
            )
        )

        # Add compression annotation
        if data.raw_tokens > 0:
            ratio = (1 - data.final_tokens / data.raw_tokens) * 100
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Compression: {ratio:.1f}%",
                showarrow=False,
                bgcolor="wheat",
            )

        title = "Token Reduction" if self.config.include_title else None
        fig.update_layout(
            title=title,
            yaxis_title="Tokens",
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
        )

        return self._save_figure(fig, output_path)

    def accuracy_vs_scale(self, data: AccuracyScaleData, output_path: Path) -> Path:
        """Generate interactive accuracy vs scale plot."""
        go, px = self._get_plotly()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.scale_points,
                y=data.baseline_accuracy,
                mode="lines+markers",
                name=data.baseline_label,
                line={"color": "#F44336", "width": 2},
                marker={"size": 10},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.scale_points,
                y=data.treatment_accuracy,
                mode="lines+markers",
                name=data.treatment_label,
                line={"color": "#4CAF50", "width": 2},
                marker={"size": 10, "symbol": "square"},
            )
        )

        # Add 90% threshold line
        fig.add_hline(y=0.9, line_dash="dash", line_color="gray", opacity=0.5)

        title = "Accuracy vs Data Scale" if self.config.include_title else None
        fig.update_layout(
            title=title,
            xaxis_title="Data Size",
            yaxis_title="Accuracy",
            xaxis_type="log",
            yaxis_range=[0, 1.05],
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.include_legend,
        )

        return self._save_figure(fig, output_path)

    def pareto_frontier(self, data: ParetoData, output_path: Path) -> Path:
        """Generate interactive Pareto frontier scatter plot."""
        go, px = self._get_plotly()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.baseline_tokens,
                y=data.baseline_accuracy,
                mode="markers",
                name="Baseline",
                marker={"color": "#F44336", "size": 12},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.treatment_tokens,
                y=data.treatment_accuracy,
                mode="markers",
                name="Treatment",
                marker={"color": "#4CAF50", "size": 12, "symbol": "square"},
            )
        )

        title = "Pareto Frontier: Tokens vs Accuracy" if self.config.include_title else None
        fig.update_layout(
            title=title,
            xaxis_title="Tokens",
            yaxis_title="Accuracy",
            xaxis_type="log",
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.include_legend,
        )

        return self._save_figure(fig, output_path)

    def confusion_matrix_heatmap(self, data: ConfusionMatrixData, output_path: Path) -> Path:
        """Generate interactive confusion matrix heatmap."""
        go, px = self._get_plotly()

        fig = px.imshow(
            data.matrix,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            x=data.labels,
            y=data.labels,
            color_continuous_scale="Blues",
            text_auto=".0f",
        )

        title = data.title if self.config.include_title else None
        fig.update_layout(
            title=title,
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
        )

        return self._save_figure(fig, output_path)

    def error_distribution_histogram(self, data: ErrorDistributionData, output_path: Path) -> Path:
        """Generate interactive error distribution histogram."""
        go, px = self._get_plotly()

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=data.baseline_errors,
                name="Baseline",
                opacity=0.6,
                marker_color="#F44336",
                nbinsx=data.bins,
            )
        )

        fig.add_trace(
            go.Histogram(
                x=data.treatment_errors,
                name="Treatment",
                opacity=0.6,
                marker_color="#4CAF50",
                nbinsx=data.bins,
            )
        )

        # Add mean lines
        fig.add_vline(
            x=float(np.mean(data.baseline_errors)),
            line_dash="dash",
            line_color="#F44336",
            line_width=2,
        )
        fig.add_vline(
            x=float(np.mean(data.treatment_errors)),
            line_dash="dash",
            line_color="#4CAF50",
            line_width=2,
        )

        title = "Error Distribution" if self.config.include_title else None
        fig.update_layout(
            title=title,
            xaxis_title="Error",
            yaxis_title="Frequency",
            barmode="overlay",
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.include_legend,
        )

        return self._save_figure(fig, output_path)

    def robustness_heatmap(self, data: RobustnessHeatmapData, output_path: Path) -> Path:
        """Generate interactive robustness heatmap."""
        go, px = self._get_plotly()

        fig = px.imshow(
            data.accuracy_matrix,
            labels={"x": "Perturbation Level", "y": "Perturbation Type", "color": "Accuracy"},
            x=[f"{level:.2f}" for level in data.perturbation_levels],
            y=data.perturbation_types,
            color_continuous_scale="RdYlGn",
            zmin=0,
            zmax=1,
            text_auto=".2f",
        )

        title = "Robustness Heatmap" if self.config.include_title else None
        fig.update_layout(
            title=title,
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
        )

        return self._save_figure(fig, output_path)

    def comparison_bar_chart(self, data: ComparisonBarData, output_path: Path) -> Path:
        """Generate interactive comparison bar chart."""
        go, px = self._get_plotly()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=data.categories,
                y=data.baseline_values,
                name="Baseline",
                marker_color="#F44336",
                text=[f"{v:.2f}" for v in data.baseline_values],
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                x=data.categories,
                y=data.treatment_values,
                name="Treatment",
                marker_color="#4CAF50",
                text=[f"{v:.2f}" for v in data.treatment_values],
                textposition="outside",
            )
        )

        title = f"{data.metric_name} Comparison" if self.config.include_title else None
        fig.update_layout(
            title=title,
            yaxis_title=data.metric_name,
            barmode="group",
            template=self._get_template(),
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.include_legend,
        )

        return self._save_figure(fig, output_path)


class BenchmarkVisualizer:
    """
    High-level visualization interface for benchmark results.

    Supports dual backends (matplotlib and plotly) with automatic fallback.
    """

    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()
        self._backend: VisualizationBackend | None = None

    @property
    def backend(self) -> VisualizationBackend:
        """Get the visualization backend (lazy initialization)."""
        if self._backend is None:
            self._backend = self._create_backend()
        return self._backend

    def _create_backend(self) -> VisualizationBackend:
        """Create the appropriate backend based on config."""
        backend: VisualizationBackend
        if self.config.backend == "plotly":
            plotly_backend = PlotlyBackend(self.config)
            if not plotly_backend.is_available():
                logger.warning("plotly not available, falling back to matplotlib")
                backend = MatplotlibBackend(self.config)
            else:
                backend = plotly_backend
        else:
            backend = MatplotlibBackend(self.config)

        if not backend.is_available():
            raise ImportError(
                "No visualization backend available. Install with: pip install semantic-frame[viz]"
            )

        return backend

    def token_reduction_waterfall(self, data: TokenReductionData, output_path: Path | str) -> Path:
        """Generate token reduction waterfall chart."""
        return self.backend.token_reduction_waterfall(data, Path(output_path))

    def accuracy_vs_scale(self, data: AccuracyScaleData, output_path: Path | str) -> Path:
        """Generate accuracy vs scale plot."""
        return self.backend.accuracy_vs_scale(data, Path(output_path))

    def pareto_frontier(self, data: ParetoData, output_path: Path | str) -> Path:
        """Generate Pareto frontier plot."""
        return self.backend.pareto_frontier(data, Path(output_path))

    def confusion_matrix_heatmap(self, data: ConfusionMatrixData, output_path: Path | str) -> Path:
        """Generate confusion matrix heatmap."""
        return self.backend.confusion_matrix_heatmap(data, Path(output_path))

    def error_distribution_histogram(
        self, data: ErrorDistributionData, output_path: Path | str
    ) -> Path:
        """Generate error distribution histogram."""
        return self.backend.error_distribution_histogram(data, Path(output_path))

    def robustness_heatmap(self, data: RobustnessHeatmapData, output_path: Path | str) -> Path:
        """Generate robustness heatmap."""
        return self.backend.robustness_heatmap(data, Path(output_path))

    def comparison_bar_chart(self, data: ComparisonBarData, output_path: Path | str) -> Path:
        """Generate comparison bar chart."""
        return self.backend.comparison_bar_chart(data, Path(output_path))

    def generate_all(
        self,
        output_dir: Path | str,
        token_data: TokenReductionData | None = None,
        scale_data: AccuracyScaleData | None = None,
        pareto_data: ParetoData | None = None,
        confusion_data: ConfusionMatrixData | None = None,
        error_data: ErrorDistributionData | None = None,
        robustness_data: RobustnessHeatmapData | None = None,
        comparison_data: ComparisonBarData | None = None,
    ) -> list[Path]:
        """
        Generate all provided visualizations.

        Args:
            output_dir: Directory to save visualizations
            *_data: Data for each chart type (optional)

        Returns:
            List of paths to generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated: list[Path] = []

        if token_data:
            path = self.token_reduction_waterfall(token_data, output_dir / "token_reduction")
            generated.append(path)

        if scale_data:
            path = self.accuracy_vs_scale(scale_data, output_dir / "accuracy_vs_scale")
            generated.append(path)

        if pareto_data:
            path = self.pareto_frontier(pareto_data, output_dir / "pareto_frontier")
            generated.append(path)

        if confusion_data:
            path = self.confusion_matrix_heatmap(confusion_data, output_dir / "confusion_matrix")
            generated.append(path)

        if error_data:
            path = self.error_distribution_histogram(error_data, output_dir / "error_distribution")
            generated.append(path)

        if robustness_data:
            path = self.robustness_heatmap(robustness_data, output_dir / "robustness_heatmap")
            generated.append(path)

        if comparison_data:
            path = self.comparison_bar_chart(comparison_data, output_dir / "comparison_bar")
            generated.append(path)

        return generated


def create_visualizer(
    backend: str = "matplotlib",
    output_format: str = "png",
    theme: str = "default",
) -> BenchmarkVisualizer:
    """
    Factory function to create a visualizer with common settings.

    Args:
        backend: "matplotlib" or "plotly"
        output_format: "png", "svg", "html", "pdf"
        theme: "default", "dark", "paper"

    Returns:
        Configured BenchmarkVisualizer
    """
    config = VisualizationConfig(
        backend=backend,
        output_format=output_format,
        theme=theme,
    )
    return BenchmarkVisualizer(config)
