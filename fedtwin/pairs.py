"""Pair construction placeholder module.

The current executable benchmark generates pairwise feature rows directly in
``data.py``. This module is kept as a stable extension point for Tier 1/Tier 2
real feature stores, where candidate-pair construction will be separated from
feature extraction.
"""

from __future__ import annotations


def pair_module_ready() -> bool:
    return True
