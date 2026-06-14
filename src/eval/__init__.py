"""Lens-based patch attribution for vision transformers.

Two evaluations, each driven by a YAML config (see ``configs/eval/``):

- :func:`eval.faithfulness.run` — insertion / deletion AUC.
- :func:`eval.localization.run` — CAAP Pointing-Game / AUPR.

Both share the model + lens-checkpoint plumbing in :mod:`eval.common`.
"""

from .config import FaithfulnessConfig, LocalizationConfig
from .faithfulness import run as run_faithfulness
from .localization import run as run_localization

__all__ = [
    "FaithfulnessConfig",
    "LocalizationConfig",
    "run_faithfulness",
    "run_localization",
]
