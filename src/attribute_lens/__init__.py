"""Lens-based patch attribution for vision transformers."""

from .config import AttributionConfig, ModelSection, LensSection, EvalSection
from .scorer import CLSLensScorer, PatchLensScorer, load_lens_checkpoint
from .metrics import insertion_curve, deletion_curve, apply_gaussian_blur
from .visualize import (
    plot_heatmap,
    plot_heatmaps_grid,
    plot_curves,
    plot_combined_report,
    plot_aggregate_curves,
)

__all__ = [
    "AttributionConfig",
    "ModelSection",
    "LensSection",
    "EvalSection",
    "CLSLensScorer",
    "PatchLensScorer",
    "load_lens_checkpoint",
    "insertion_curve",
    "deletion_curve",
    "apply_gaussian_blur",
    "plot_heatmap",
    "plot_heatmaps_grid",
    "plot_curves",
    "plot_combined_report",
    "plot_aggregate_curves",
]
