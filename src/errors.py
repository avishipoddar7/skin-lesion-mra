"""Exceptions raised by pipeline stages; only the app turns them into UI messages."""


class InvalidImageError(ValueError):
    """The input file is not a usable dermoscopy image."""


class NoLesionError(RuntimeError):
    """Segmentation found no plausible lesion."""
