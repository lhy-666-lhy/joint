"""Apply CUDA-ops fallback before importing Point-M2AE modules."""

from __future__ import annotations

_APPLIED = False


def ensure_ops() -> None:
    """Install pure-PyTorch stubs for pointnet2_ops / knn_cuda if missing."""
    global _APPLIED
    if _APPLIED:
        return
    try:
        import knn_cuda  # noqa: F401
        import pointnet2_ops  # noqa: F401
    except ModuleNotFoundError:
        from . import ops_fallback

        ops_fallback.apply()
    _APPLIED = True


# Auto-apply on import so `from vendor.point_m2ae import bootstrap` is enough.
ensure_ops()
