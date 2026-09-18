"""Compatibility import for the Reddit Arctic Shift collector."""

import sys

try:
    from src.platforms.reddit import collector as _collector
except ModuleNotFoundError:
    from platforms.reddit import collector as _collector

sys.modules[__name__] = _collector
