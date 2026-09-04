"""Geometric metrics for corresponding planar centerline samples."""

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class DeformationMetrics(TypedDict):
    mean_displacement: float
    max_displacement: float
    endpoint_displacement: float
    shape_rmse: float


def _validate_points(points: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (n, 2) with n >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def arc_length(points: ArrayLike) -> float:
    """Calculate polyline arc length."""
    array = _validate_points(points, "points")
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def relative_arc_length_consistency_error(points: ArrayLike, prescribed_length: float) -> float:
    """Compare sampled polyline length with the prescribed physical length.

    This is a numerical discretization check, not an extensional strain or a
    general deformation metric. The centerline model prescribes a fixed
    physical length and does not model axial extension.
    """
    if not np.isfinite(prescribed_length) or prescribed_length <= 0:
        raise ValueError("prescribed_length must be a positive finite value")
    return float(abs(arc_length(points) - prescribed_length) / prescribed_length)


def pointwise_displacement(reference: ArrayLike, deformed: ArrayLike) -> FloatArray:
    """Return Euclidean displacement at corresponding material points."""
    reference_array = _validate_points(reference, "reference")
    deformed_array = _validate_points(deformed, "deformed")
    if reference_array.shape != deformed_array.shape:
        raise ValueError("reference and deformed centerlines must have matching shapes")
    return np.linalg.norm(deformed_array - reference_array, axis=1)


def curvature_rmse(reference_curvature: ArrayLike, deformed_curvature: ArrayLike) -> float:
    """Return root-mean-square curvature difference at corresponding points."""
    reference = np.asarray(reference_curvature, dtype=float)
    deformed = np.asarray(deformed_curvature, dtype=float)
    if reference.ndim != 1 or reference.shape != deformed.shape or reference.size < 2:
        raise ValueError("curvature arrays must be one-dimensional and have matching shapes")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(deformed)):
        raise ValueError("curvature arrays must contain only finite values")
    return float(np.sqrt(np.mean(np.square(deformed - reference))))


def deformation_metrics(reference: ArrayLike, deformed: ArrayLike) -> DeformationMetrics:
    """Summarize deformation between corresponding centerline samples.

    Shape RMSE is ``sqrt(mean(||x_i - X0_i||^2))``. Arc-length differences
    are intentionally excluded because the model does not represent axial
    extension.
    """
    reference_array = _validate_points(reference, "reference")
    deformed_array = _validate_points(deformed, "deformed")
    displacement = pointwise_displacement(reference_array, deformed_array)
    return {
        "mean_displacement": float(displacement.mean()),
        "max_displacement": float(displacement.max()),
        "endpoint_displacement": float(displacement[-1]),
        "shape_rmse": float(np.sqrt(np.mean(np.square(displacement)))),
    }
