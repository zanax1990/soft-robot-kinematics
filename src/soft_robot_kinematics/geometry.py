"""Geometry utilities for planar inextensible centerlines."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
CurvatureInput = float | ArrayLike | Callable[[FloatArray], ArrayLike]


def material_coordinate(num_points: int = 201) -> FloatArray:
    """Return a uniformly sampled normalized material coordinate in [0, 1]."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    return np.linspace(0.0, 1.0, num_points, dtype=float)


def _validate_coordinate(s: ArrayLike) -> FloatArray:
    coordinate = np.asarray(s, dtype=float)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("s must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("s must contain only finite values")
    if np.any(np.diff(coordinate) <= 0):
        raise ValueError("s must be strictly increasing")
    return coordinate


def _evaluate_curvature(s: FloatArray, curvature: CurvatureInput) -> FloatArray:
    values = curvature(s) if callable(curvature) else curvature
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full_like(s, float(array))
    if array.shape != s.shape:
        raise ValueError("curvature must be scalar or have the same shape as s")
    if not np.all(np.isfinite(array)):
        raise ValueError("curvature must contain only finite values")
    return array


def _cumulative_trapezoid(values: FloatArray, spacing: FloatArray) -> FloatArray:
    increments = 0.5 * (values[1:] + values[:-1]) * spacing
    return np.concatenate(([0.0], np.cumsum(increments)))


def reconstruct_centerline(
    s: ArrayLike,
    curvature: CurvatureInput,
    *,
    length: float = 1.0,
    initial_position: tuple[float, float] = (0.0, 0.0),
    initial_angle: float = 0.0,
) -> FloatArray:
    """Reconstruct a planar centerline from curvature by tangent integration.

    ``s`` is a normalized material coordinate. Curvature is expressed per unit
    physical length, so the integration increment is ``length * ds``.
    """
    coordinate = _validate_coordinate(s)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a positive finite number")
    origin = np.asarray(initial_position, dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("initial_position must contain two finite values")
    if not np.isfinite(initial_angle):
        raise ValueError("initial_angle must be finite")

    kappa = _evaluate_curvature(coordinate, curvature)
    physical_spacing = length * np.diff(coordinate)
    tangent_angle = initial_angle + _cumulative_trapezoid(kappa, physical_spacing)
    tangent = np.column_stack((np.cos(tangent_angle), np.sin(tangent_angle)))

    segment_vectors = 0.5 * (tangent[1:] + tangent[:-1]) * physical_spacing[:, None]
    return origin + np.vstack((np.zeros(2), np.cumsum(segment_vectors, axis=0)))


def straight_centerline(
    s: ArrayLike,
    *,
    length: float = 1.0,
    initial_position: tuple[float, float] = (0.0, 0.0),
    initial_angle: float = 0.0,
) -> FloatArray:
    """Return a straight reference centerline ``X0(s)``."""
    return reconstruct_centerline(
        s,
        0.0,
        length=length,
        initial_position=initial_position,
        initial_angle=initial_angle,
    )


def constant_curvature(s: ArrayLike, value: float) -> FloatArray:
    """Return a constant curvature field sampled on ``s``."""
    coordinate = _validate_coordinate(s)
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return np.full_like(coordinate, value)


def varying_curvature(
    s: ArrayLike,
    *,
    amplitude: float,
    offset: float = 0.0,
    cycles: float = 1.0,
) -> FloatArray:
    """Return a smooth sinusoidal curvature field for synthetic experiments."""
    coordinate = _validate_coordinate(s)
    parameters = np.asarray([amplitude, offset, cycles], dtype=float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("curvature parameters must be finite")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    phase = (coordinate - coordinate[0]) / (coordinate[-1] - coordinate[0])
    return offset + amplitude * np.sin(2.0 * np.pi * cycles * phase)
