"""test_shape_to_wavelength.py

Unit tests for needle_shape_publisher.shape_to_wavelength.

These tests verify the roundtrip property:
    shape  →  Δλ  →  curvature  ≈  expected curvature from shape.

Run with pytest (no ROS2 environment required):

    cd needle_shape_publisher
    python -m pytest test/test_shape_to_wavelength.py -v
"""

import json
import os
import tempfile

import numpy as np
import pytest

from needle_shape_publisher.shape_to_wavelength import (
    MM_TO_M_CURVATURE_SCALE,
    build_parallel_transport_frame,
    compute_arc_lengths,
    compute_curvature_components,
    compute_tangents,
    curvatures_to_wavelength_shifts,
    interpolate_curvature_at_sensors,
    load_needle_params_json,
    load_shape_file,
    shape_to_wavelength_shifts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_NEEDLE_DATA = os.path.join(
    _HERE, '..', 'needle_data', '3CH-4AA-0005',
    '3CH-4AA-0005_needle_params_2022-01-26_Jig-Calibration_best_weights.json',
)
_DEFAULT_SHAPE_YAML = os.path.join(
    _HERE, '..', 'needle_data', 'sim_shape_default.yaml',
)


def make_circular_arc(kappa, L=100.0, n=51):
    """Generate a planar circular arc in the x-z plane.

    Returns (N,3) array of points, one-dimensional curvature = kappa (rad/mm).
    The arc starts at the origin pointing along +z and curves toward +x.
    """
    R = 1.0 / kappa
    s = np.linspace(0, L, n)
    x = R * (1.0 - np.cos(s / R))
    y = np.zeros(n)
    z = R * np.sin(s / R)
    return np.column_stack([x, y, z])


def make_yz_circular_arc(kappa, L=100.0, n=51):
    """Generate a planar circular arc in the y-z plane curving toward -y.

    Returns (N,3) array of points, one-dimensional curvature = kappa (rad/mm).
    The arc starts at the origin pointing along +z and curves toward -y.
    """
    R = 1.0 / kappa
    s = np.linspace(0, L, n)
    x = np.zeros(n)
    y = -R * (1.0 - np.cos(s / R))
    z = R * np.sin(s / R)
    return np.column_stack([x, y, z])


def make_test_needle_params_json(tmp_path, num_chs=3, num_aas=2):
    """Write a minimal needle param JSON and return its path.

    Sensor locations are stored as mm from the needle TIP (matching the
    convention expected by ``load_needle_params_json``).
    """
    cal_mats = {}
    sensor_locs = {}
    for i in range(1, num_aas + 1):
        loc = float(i * 20)  # 20 mm, 40 mm, ... from tip
        sensor_locs[str(i)] = loc
        # 2 x num_chs identity-like matrix: [ω[0], ω[1]] = C @ Δλ
        # Use first two channels: C[:, 0] = [1,0], C[:, 1] = [0,1], rest 0
        row = np.zeros((2, num_chs))
        row[0, 0] = 1.0
        row[1, 1] = 1.0
        cal_mats[str(loc)] = row.tolist()

    params = {
        'serial number': 'TEST',
        'length': 200.0,
        'diameter': 1.27,
        'Emod': 200000,
        'pratio': 0.29,
        '# channels': num_chs,
        '# active areas': num_aas,
        'Sensor Locations': sensor_locs,
        'Calibration Matrices': cal_mats,
        'weights': {str(float(i * 20)): 1.0 / num_aas for i in range(1, num_aas + 1)},
    }

    fpath = str(tmp_path / 'test_needle_params.json')
    with open(fpath, 'w') as fh:
        json.dump(params, fh)
    return fpath


# ---------------------------------------------------------------------------
# Tests: arc-length and tangent computation
# ---------------------------------------------------------------------------

class TestArcLengths:
    def test_straight_needle_arc_length(self):
        """Arc length of a straight needle along z equals its length."""
        n = 11
        L = 100.0
        points = np.column_stack([np.zeros(n), np.zeros(n), np.linspace(0, L, n)])
        s = compute_arc_lengths(points)
        assert s[0] == pytest.approx(0.0)
        assert s[-1] == pytest.approx(L)

    def test_circular_arc_length(self):
        """Arc length of a circular arc should equal s_max."""
        kappa = 0.003
        L = 100.0
        pts = make_circular_arc(kappa, L=L, n=201)
        s = compute_arc_lengths(pts)
        assert s[-1] == pytest.approx(L, rel=1e-3)


class TestTangents:
    def test_straight_tangents_along_z(self):
        """Tangents of a straight z-axis needle are all [0, 0, 1]."""
        n = 11
        L = 100.0
        points = np.column_stack([
            np.zeros(n), np.zeros(n), np.linspace(0, L, n)
        ])
        s = compute_arc_lengths(points)
        tangents = compute_tangents(points, s)
        expected = np.tile([0, 0, 1], (n, 1))
        np.testing.assert_allclose(tangents, expected, atol=1e-6)

    def test_tangents_unit_norm(self):
        """All tangent vectors must be unit-normalised."""
        kappa = 0.003
        pts = make_circular_arc(kappa, L=100.0, n=51)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        norms = np.linalg.norm(tangents, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: parallel-transport frame
# ---------------------------------------------------------------------------

class TestParallelTransportFrame:
    def test_frame_orthonormality(self):
        """Each frame must be orthonormal."""
        kappa = 0.003
        pts = make_circular_arc(kappa, L=100.0, n=51)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)

        for i in range(len(frames)):
            F = frames[i]  # 3x3 with rows [d1, d2, d3]
            prod = F @ F.T
            np.testing.assert_allclose(prod, np.eye(3), atol=1e-8,
                                       err_msg=f'Frame {i} not orthonormal')

    def test_frame_d3_equals_tangent(self):
        """Row 2 (d3) of each frame must equal the tangent vector."""
        kappa = 0.003
        pts = make_circular_arc(kappa, L=100.0, n=21)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        np.testing.assert_allclose(frames[:, 2, :], tangents, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: curvature computation for a circular arc
# ---------------------------------------------------------------------------

class TestCurvatureComputation:
    def test_planar_arc_curvature(self):
        """A planar arc in x-z has constant curvature magnitude ≈ kappa, κy≈0."""
        kappa = 0.003
        L = 100.0
        pts = make_circular_arc(kappa, L=L, n=101)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)

        # Curvature magnitude must equal kappa regardless of frame orientation.
        # (d1 is initialised to the y-axis for a z-directed initial tangent, so
        #  the x-z bending shows up in ky; checking the magnitude is robust.)
        kappa_mag = np.sqrt(kx ** 2 + ky ** 2)
        # Ignore endpoints where boundary effects from the finite-diff / SavGol
        # filter are largest.
        interior = slice(5, -5)
        np.testing.assert_allclose(
            kappa_mag[interior],
            np.full(len(kappa_mag[interior]), kappa),
            rtol=0.05,
            err_msg='Curvature magnitude should be ~kappa for a circular arc'
        )


# ---------------------------------------------------------------------------
# Tests: body-frame convention (kx, ky) → (ω[0], ω[1]) = (−κy, +κx)
# ---------------------------------------------------------------------------

class TestBodyFrameConvention:
    """Verify that interpolate_curvature_at_sensors maps (kx, ky) to (−ky, kx).

    Body-frame derivation:
        dR/ds = R · skew(ω)  →  dT/ds = ω[1]·e1 − ω[0]·e2
        projecting: kx = (dT/ds)·e1 = ω[1],  ky = (dT/ds)·e2 = −ω[0]
        ⟹  ω[0] = −ky,   ω[1] = +kx
    """

    def test_xz_arc_omega_slots(self):
        """X-Z plane arc (bends toward +x): kx≈kappa, ky≈0.

        After the body-frame mapping:
            kappa_at[:, 0] = ω[0] = −ky ≈  0
            kappa_at[:, 1] = ω[1] = +kx ≈ kappa
        """
        kappa_val = 0.003
        L = 100.0
        pts = make_circular_arc(kappa_val, L=L, n=101)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)

        # Sensor at mid-arc where boundary effects are minimal
        s_mid = [L / 2.0]
        kappa_at = interpolate_curvature_at_sensors(s, kx, ky, s_mid)

        tol = kappa_val * 0.10  # 10 % relative tolerance
        np.testing.assert_allclose(
            kappa_at[0, 0], 0.0, atol=tol,
            err_msg='ω[0] should be ≈0 for an x-z plane arc (ky≈0 → −ky≈0)'
        )
        np.testing.assert_allclose(
            kappa_at[0, 1], kappa_val, rtol=0.10,
            err_msg='ω[1] should be ≈kappa for an x-z plane arc (kx≈kappa)'
        )

    def test_yz_arc_omega_slots(self):
        """Y-Z plane arc (bends toward −y): kx≈0, ky≈−kappa (ky<0 because n·d2<0).

        After the body-frame mapping:
            kappa_at[:, 0] = ω[0] = −ky ≈ +kappa
            kappa_at[:, 1] = ω[1] = +kx ≈  0
        """
        kappa_val = 0.003
        L = 100.0
        pts = make_yz_circular_arc(kappa_val, L=L, n=101)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)

        s_mid = [L / 2.0]
        kappa_at = interpolate_curvature_at_sensors(s, kx, ky, s_mid)

        tol = kappa_val * 0.10
        np.testing.assert_allclose(
            kappa_at[0, 0], kappa_val, rtol=0.10,
            err_msg='ω[0] should be ≈kappa for a y-z plane arc (ky≈−kappa → −ky≈+kappa)'
        )
        np.testing.assert_allclose(
            kappa_at[0, 1], 0.0, atol=tol,
            err_msg='ω[1] should be ≈0 for a y-z plane arc (kx≈0)'
        )

    def test_piecewise_exp_reconstruction_xz_arc(self, tmp_path):
        """Roundtrip: x-z arc → Δλ → C @ Δλ gives ω = [0, kappa], not [kappa, 0]."""
        param_file = make_test_needle_params_json(tmp_path, num_chs=3, num_aas=2)
        (cal_matrices, sensor_locs_from_tip,
         num_chs, num_aas, _) = load_needle_params_json(param_file)

        kappa = 0.005
        L = 80.0
        pts = make_circular_arc(kappa, L=L, n=81)
        insertion_depth = L

        delta_lambda = shape_to_wavelength_shifts(
            pts, param_file, insertion_depth=insertion_depth
        )

        sensor_locs_from_base = [
            max(0.0, insertion_depth - loc) for loc in sensor_locs_from_tip
        ]

        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx_arr, ky_arr = compute_curvature_components(s, tangents, frames)
        kappa_at = interpolate_curvature_at_sensors(
            s, kx_arr, ky_arr, sensor_locs_from_base
        )

        for aa_idx, loc in enumerate(sensor_locs_from_tip):
            C = cal_matrices[float(loc)]
            dl = np.array([delta_lambda[ch][aa_idx] for ch in range(1, num_chs + 1)])
            omega_processed = C @ dl
            omega_reconstructed = omega_processed / MM_TO_M_CURVATURE_SCALE

            np.testing.assert_allclose(
                omega_reconstructed, kappa_at[aa_idx],
                rtol=1e-6,
                err_msg=(
                    f'Roundtrip failed at AA {aa_idx}: '
                    f'expected ω={kappa_at[aa_idx]}, got {omega_reconstructed}'
                )
            )

            # For x-z arc, slot 0 (ω[0]) should be ≈0 and slot 1 (ω[1]) ≈ kappa
            np.testing.assert_allclose(
                omega_reconstructed[0], 0.0, atol=kappa * 0.12,
                err_msg='ω[0] should be ≈0 for x-z arc (not kappa!)'
            )
            np.testing.assert_allclose(
                omega_reconstructed[1], kappa, rtol=0.12,
                err_msg='ω[1] should be ≈kappa for x-z arc'
            )


# ---------------------------------------------------------------------------
# Tests: wavelength shift inversion
# ---------------------------------------------------------------------------

class TestWavelengthShiftInversion:
    """Roundtrip test: shape → Δλ → curvature ≈ expected."""

    def test_roundtrip_with_identity_cal_matrix(self, tmp_path):
        """With identity calibration, C @ Δλ ≈ kappa_at at each sensor."""
        param_file = make_test_needle_params_json(tmp_path, num_chs=3, num_aas=2)
        (cal_matrices, sensor_locs_from_tip,
         num_chs, num_aas, _) = load_needle_params_json(param_file)

        kappa = 0.005
        L = 80.0
        pts = make_circular_arc(kappa, L=L, n=81)
        insertion_depth = L

        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)

        sensor_locs_from_base = [
            max(0.0, insertion_depth - loc) for loc in sensor_locs_from_tip
        ]
        kappa_at = interpolate_curvature_at_sensors(
            s, kx, ky, sensor_locs_from_base
        )

        delta_lambda = curvatures_to_wavelength_shifts(
            kappa_at * MM_TO_M_CURVATURE_SCALE,
            cal_matrices,
            sensor_locs_from_tip,
            num_chs,
        )

        # Forward pass: reconstruct curvature from delta_lambda
        for aa_idx, loc in enumerate(sensor_locs_from_tip):
            C = cal_matrices[float(loc)]
            dl = np.array([delta_lambda[ch][aa_idx] for ch in range(1, num_chs + 1)])
            kappa_processed = C @ dl
            kappa_reconstructed = kappa_processed / MM_TO_M_CURVATURE_SCALE
            np.testing.assert_allclose(
                kappa_reconstructed, kappa_at[aa_idx],
                rtol=1e-6,
                err_msg=f'Roundtrip failed at AA {aa_idx} (loc_from_tip={loc})'
            )

    def test_roundtrip_with_real_needle_params(self):
        """Roundtrip with the default 3CH-4AA-0005 needle param file."""
        if not os.path.isfile(_NEEDLE_DATA):
            pytest.skip('Needle param file not found; skipping integration test')

        (cal_matrices, sensor_locs_from_tip,
         num_chs, num_aas, _) = load_needle_params_json(_NEEDLE_DATA)

        kappa = 0.003
        L = 120.0
        pts = make_circular_arc(kappa, L=L, n=121)
        insertion_depth = 120.0

        delta_lambda = shape_to_wavelength_shifts(
            pts, _NEEDLE_DATA, insertion_depth=insertion_depth
        )

        # Verify roundtrip: C @ Δλ ≈ kappa_expected (body-frame)
        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)
        sensor_locs_from_base = [
            max(0.0, insertion_depth - loc) for loc in sensor_locs_from_tip
        ]
        kappa_at = interpolate_curvature_at_sensors(
            s, kx, ky, sensor_locs_from_base
        )

        for aa_idx, loc in enumerate(sensor_locs_from_tip):
            C = cal_matrices[float(loc)]
            dl = np.array([delta_lambda[ch][aa_idx] for ch in range(1, num_chs + 1)])
            kappa_processed = C @ dl
            kappa_reconstructed = kappa_processed / MM_TO_M_CURVATURE_SCALE
            np.testing.assert_allclose(
                kappa_reconstructed, kappa_at[aa_idx],
                rtol=1e-5,
                atol=1e-12,
                err_msg=f'Roundtrip failed at sensor loc {loc} mm from tip'
            )


class TestSimulationCurvatureUnits:
    """Verify the sim bridge preserves mm-based curvatures after calibration."""

    def test_default_sim_shape_roundtrips_to_mm_curvature(self):
        """Default sim YAML should still publish/log curvatures in rad/mm."""
        if not os.path.isfile(_DEFAULT_SHAPE_YAML) or not os.path.isfile(_NEEDLE_DATA):
            pytest.skip('Default shape YAML or needle param file not found')

        insertion_depth = 100.0
        pts = load_shape_file(_DEFAULT_SHAPE_YAML)
        (cal_matrices, sensor_locs_from_tip,
         num_chs, _, _) = load_needle_params_json(_NEEDLE_DATA)

        delta_lambda = shape_to_wavelength_shifts(
            pts, _NEEDLE_DATA, insertion_depth=insertion_depth
        )

        s = compute_arc_lengths(pts)
        tangents = compute_tangents(pts, s)
        frames = build_parallel_transport_frame(tangents)
        kx, ky = compute_curvature_components(s, tangents, frames)
        sensor_locs_from_base = [
            max(0.0, insertion_depth - loc) for loc in sensor_locs_from_tip
        ]
        kappa_expected_mm = interpolate_curvature_at_sensors(
            s, kx, ky, sensor_locs_from_base
        )

        for aa_idx, loc in enumerate(sensor_locs_from_tip):
            C = cal_matrices[float(loc)]
            dl = np.array([
                delta_lambda[ch][aa_idx] for ch in range(1, num_chs + 1)
            ])
            kappa_processed_m = C @ dl
            kappa_current_mm = kappa_processed_m / MM_TO_M_CURVATURE_SCALE

            np.testing.assert_allclose(
                kappa_current_mm,
                kappa_expected_mm[aa_idx],
                rtol=1e-5,
                atol=1e-12,
                err_msg=f'Simulated AA {aa_idx} should preserve rad/mm curvature',
            )

            np.testing.assert_allclose(
                np.linalg.norm(kappa_current_mm),
                np.linalg.norm(kappa_expected_mm[aa_idx]),
                rtol=1e-5,
                atol=1e-12,
            )

        np.testing.assert_allclose(
            np.linalg.norm(kappa_expected_mm, axis=1),
            np.full(len(sensor_locs_from_tip), 0.003),
            rtol=5e-3,
            err_msg='Default sim shape should recover ~0.003 rad/mm curvature',
        )


# ---------------------------------------------------------------------------
# Tests: load_shape_file
# ---------------------------------------------------------------------------

class TestLoadShapeFile:
    def test_load_default_yaml(self):
        """Default YAML shape file must load and have shape (N, 3)."""
        if not os.path.isfile(_DEFAULT_SHAPE_YAML):
            pytest.skip('Default shape YAML not found')
        pts = load_shape_file(_DEFAULT_SHAPE_YAML)
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert pts.shape[0] >= 2

    def test_load_json_shape(self, tmp_path):
        """JSON shape file must load correctly."""
        data = {'shape': [[0.0, 0.0, 0.0], [1.0, 0.0, 10.0]]}
        fpath = str(tmp_path / 'shape.json')
        with open(fpath, 'w') as fh:
            json.dump(data, fh)
        pts = load_shape_file(fpath)
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pts[1], [1.0, 0.0, 10.0])


# ---------------------------------------------------------------------------
# Tests: load_needle_params_json
# ---------------------------------------------------------------------------

class TestLoadNeedleParams:
    def test_load_default_needle(self):
        """Load the 3CH-4AA-0005 needle param file."""
        if not os.path.isfile(_NEEDLE_DATA):
            pytest.skip('Needle param file not found')
        (cal_matrices, sensor_locs_from_tip,
         num_chs, num_aas, needle_length) = load_needle_params_json(_NEEDLE_DATA)
        assert num_chs == 3
        assert num_aas == 4
        assert len(sensor_locs_from_tip) == num_aas
        assert needle_length > 0
        for loc, C in cal_matrices.items():
            assert C.shape == (2, num_chs), \
                f'Calibration matrix at {loc} has wrong shape {C.shape}'
