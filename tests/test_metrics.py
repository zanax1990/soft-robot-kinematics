import numpy as np
import pytest

from soft_robot_kinematics import (
    arc_length,
    constant_curvature,
    curvature_rmse,
    deformation_metrics,
    material_coordinate,
    reconstruct_centerline,
    relative_arc_length_consistency_error,
    straight_centerline,
)


def test_identity_deformation_metrics_are_zero():
    reference = straight_centerline(material_coordinate(101), length=2.5)
    metrics = deformation_metrics(reference, reference.copy())
    for name in [
        "mean_displacement",
        "max_displacement",
        "endpoint_displacement",
        "shape_rmse",
    ]:
        assert metrics[name] == pytest.approx(0.0, abs=1e-12)


def test_translation_metrics_are_consistent():
    reference = straight_centerline(material_coordinate(21))
    translated = reference + np.array([3.0, 4.0])
    metrics = deformation_metrics(reference, translated)
    assert metrics["mean_displacement"] == pytest.approx(5.0)
    assert metrics["max_displacement"] == pytest.approx(5.0)
    assert metrics["endpoint_displacement"] == pytest.approx(5.0)
    assert metrics["shape_rmse"] == pytest.approx(5.0)


def test_arc_length_of_straight_line():
    reference = straight_centerline(material_coordinate(31), length=3.2)
    assert arc_length(reference) == pytest.approx(3.2)


def test_arc_length_consistency_error_decreases_with_resolution():
    coarse_s = material_coordinate(51)
    fine_s = material_coordinate(401)
    coarse = reconstruct_centerline(
        coarse_s,
        constant_curvature(coarse_s, 4.0),
        length=1.0,
    )
    fine = reconstruct_centerline(fine_s, constant_curvature(fine_s, 4.0), length=1.0)
    coarse_error = relative_arc_length_consistency_error(coarse, 1.0)
    fine_error = relative_arc_length_consistency_error(fine, 1.0)
    assert 0.0 <= fine_error < coarse_error


def test_arc_length_consistency_error_rejects_invalid_length():
    reference = straight_centerline(material_coordinate(11))
    with pytest.raises(ValueError, match="positive finite"):
        relative_arc_length_consistency_error(reference, 0.0)


def test_curvature_rmse_for_identical_fields_is_zero():
    curvature = np.linspace(-1.0, 1.0, 25)
    assert curvature_rmse(curvature, curvature.copy()) == pytest.approx(0.0)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        deformation_metrics(np.zeros((4, 2)), np.zeros((5, 2)))
