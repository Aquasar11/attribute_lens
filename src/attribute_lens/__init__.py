"""Lens-based patch attribution for vision transformers."""

from .config import AttributionConfig, ModelSection, LensSection, EvalSection, LayerAvgSection, NeighborAvgSection
from .postprocess import (
    neighbor_avg_embeddings,
    neighbor_avg_scores,
    layer_avg_score_maps,
    load_layer_weights,
)
from .scorer import CLSLensScorer, PatchLensScorer, PatchMapCLSLensScorer, load_lens_checkpoint, load_patch_map_checkpoint
from .metrics import insertion_curve, deletion_curve, insertion_deletion_curves, insertion_deletion_curves_batch, insertion_deletion_curves_all_layers, apply_gaussian_blur
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
    "LayerAvgSection",
    "NeighborAvgSection",
    "neighbor_avg_embeddings",
    "neighbor_avg_scores",
    "layer_avg_score_maps",
    "load_layer_weights",
    "CLSLensScorer",
    "PatchLensScorer",
    "PatchMapCLSLensScorer",
    "load_lens_checkpoint",
    "load_patch_map_checkpoint",
    "insertion_curve",
    "deletion_curve",
    "insertion_deletion_curves",
    "insertion_deletion_curves_batch",
    "insertion_deletion_curves_all_layers",
    "apply_gaussian_blur",
    "plot_heatmap",
    "plot_heatmaps_grid",
    "plot_curves",
    "plot_combined_report",
    "plot_aggregate_curves",
]
