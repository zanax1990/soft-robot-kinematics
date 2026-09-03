"""Deformation metrics for corresponding planar centerline samples."""

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class DeformationMetrics(TypedDict):
    mean_displacement: float
    max_displacement: float
    endpoint_displacement: float
    shape_rmse: float
    reference_arc_length: float
    deformed_arc_length: float
    arc_length_change: float
    relative_arc_length_change: float


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

    Shape RMSE is ``sqrt(mean(||x_i - X0_i||^2))``. Arc-length change is
    signed: positive values indicate an increase in sampled polyline length.
    """
    reference_array = _validate_points(reference, "reference")
    deformed_array = _validate_points(deformed, "deformed")
    displacement = pointwise_displacement(reference_array, deformed_array)
    reference_length = arc_length(reference_array)
    deformed_length = arc_length(deformed_array)
    length_change = deformed_length - reference_length
    relative_change = length_change / reference_length if reference_length > 0 else np.nan

    return {
        "mean_displacement": float(displacement.mean()),
        "max_displacement": float(displacement.max()),
        "endpoint_displacement": float(displacement[-1]),
        "shape_rmse": float(np.sqrt(np.mean(np.square(displacement)))),
        "reference_arc_length": reference_length,
        "deformed_arc_length": deformed_length,
        "arc_length_change": float(length_change),
        "relative_arc_length_change": float(relative_change),
    }
