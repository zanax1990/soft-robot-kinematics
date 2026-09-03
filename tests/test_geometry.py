import numpy as np
import pytest

from soft_robot_kinematics import (
    constant_curvature,
    material_coordinate,
    reconstruct_centerline,
    straight_centerline,
    varying_curvature,
)


def test_material_coordinate_bounds_and_shape():
    s = material_coordinate(51)
    assert s.shape == (51,)
    assert s[0] == 0.0
    assert s[-1] == 1.0


def test_zero_curvature_produces_straight_centerline():
    s = material_coordinate(101)
    centerline = reconstruct_centerline(s, 0.0, length=2.0)
    np.testing.assert_allclose(centerline[:, 0], 2.0 * s, atol=1e-12)
    np.testing.assert_allclose(centerline[:, 1], 0.0, atol=1e-12)


def test_straight_centerline_respects_initial_pose():
    s = material_coordinate(11)
    centerline = straight_centerline(
        s,
        length=1.5,
        initial_position=(1.0, -2.0),
        initial_angle=np.pi / 2,
    )
    np.testing.assert_allclose(centerline[:, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(centerline[:, 1], -2.0 + 1.5 * s, atol=1e-12)


def test_constant_curvature_matches_circular_arc_endpoint():
    s = material_coordinate(4001)
    kappa = 2.0
    length = 0.8
    centerline = reconstruct_centerline(s, constant_curvature(s, kappa), length=length)
    expected = np.array([np.sin(kappa * length) / kappa, (1 - np.cos(kappa * length)) / kappa])
    np.testing.assert_allclose(centerline[-1], expected, atol=2e-7)


def test_varying_curvature_is_finite_and_has_expected_shape():
    s = material_coordinate(73)
    kappa = varying_curvature(s, amplitude=3.0, offset=0.5)
    centerline = reconstruct_centerline(s, kappa)
    assert kappa.shape == s.shape
    assert centerline.shape == (73, 2)
    assert np.isfinite(centerline).all()


def test_rejects_non_increasing_coordinate():
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_centerline([0.0, 0.5, 0.5], 0.0)
