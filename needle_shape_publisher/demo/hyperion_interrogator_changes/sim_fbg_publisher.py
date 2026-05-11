"""sim_fbg_publisher.py  –  to be placed in hyperion_interrogator/

Shape-driven FBG wavelength-shift generator for sim_level=1 simulation.

Given a 3-D needle-shape polyline (loaded from a YAML or JSON file), computes
per-channel FBG wavelength shifts (Δλ) such that, when processed through the
existing calibration pipeline, the inferred curvature matches the curvature of
the supplied shape.

Pipeline
--------
shape polyline  (Nx3, mm)
  → arc-length parametrisation
  → unit tangents  (Savitzky-Golay local-quadratic or central differences)
  → rotation-minimising parallel-transport frame
  → curvature components (κx, κy) in the material frame
  → body-frame angular-velocity mapping: ω[0] = −κy, ω[1] = +κx
  → interpolation at each active-area (AA) sensor location
  → inverse calibration:  Δλ_k = pinv(C_k) @ [ω[0]_k, ω[1]_k]

Body-frame convention
---------------------
For rotation matrix R with columns [e1, e2, e3=tangent]:

    dR/ds = R · skew(ω)   →   dT/ds = ω[1]·e1 − ω[0]·e2

Projecting:
    κx = (dT/ds)·e1 = +ω[1]   →   ω[1] = +κx
    κy = (dT/ds)·e2 = −ω[0]   →   ω[0] = −κy

This is the convention stored in ``current_curvatures[0:2, aa]`` after the
normal FBG processing pipeline.

References
----------
Conceptual baseline: ``piecewise_exp_test.py``, ``piecewise_exp`` branch,
``rban01/needle_shape_sensing_original`` – Savgol tangent estimation +
parallel-transport frame + curvature extraction.
"""

import json
import os

import numpy as np

# Optional scipy
try:
    from scipy.signal import savgol_filter as _savgol
    _SCIPY = True
except ImportError:
    _SCIPY = False

# Optional PyYAML
try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False


# ---------------------------------------------------------------------------
# Shape file loading
# ---------------------------------------------------------------------------

def load_shape_file(shape_file: str) -> np.ndarray:
    """Load a 3-D shape polyline from a YAML or JSON file.

    Expected format (YAML example)::

        shape:
          - [x0, y0, z0]
          - [x1, y1, z1]
          ...

    The top-level key may be ``shape`` or ``points``.

    Returns
    -------
    np.ndarray, shape (N, 3), units mm
    """
    ext = os.path.splitext(shape_file)[1].lower()
    if ext in ('.yaml', '.yml'):
        if not _YAML:
            raise ImportError(
                "PyYAML is required for .yaml shape files. "
                "Install it with: pip install PyYAML"
            )
        with open(shape_file, 'r') as fh:
            data = _yaml.safe_load(fh)
    else:
        with open(shape_file, 'r') as fh:
            data = json.load(fh)

    key = 'shape' if 'shape' in data else 'points'
    pts = np.asarray(data[key], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"Shape file must contain an Nx3 array; got {pts.shape}"
        )
    return pts


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc lengths along a polyline."""
    return np.concatenate([[0.0],
                           np.cumsum(np.linalg.norm(np.diff(points, axis=0),
                                                    axis=1))])


def _deriv_cols(data: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Column-wise first derivative (SavGol if available, else gradient)."""
    n = data.shape[0]
    if _SCIPY and n >= 5:
        w = min(5, n)
        if w % 2 == 0:
            w -= 1
        delta = float(np.mean(np.diff(arc))) if n > 1 else 1.0
        return np.column_stack([
            _savgol(data[:, i], window_length=w, polyorder=2,
                    deriv=1, delta=delta)
            for i in range(data.shape[1])
        ])
    return np.gradient(data, arc, axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-wise unit normalisation (safe, no division by zero)."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms < 1e-10, 1.0, norms)


def _tangents(points: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Unit tangent vectors via SavGol or central differences."""
    return _unit(_deriv_cols(points, arc))


def _parallel_transport_frames(tangents: np.ndarray) -> np.ndarray:
    """Rotation-minimising frames.  Returns (N,3,3) where rows = [d1,d2,d3]."""
    n = len(tangents)
    F = np.zeros((n, 3, 3))

    t0 = tangents[0]
    ax = int(np.argmin(np.abs(t0)))
    e = np.zeros(3); e[ax] = 1.0
    d1 = e - np.dot(e, t0) * t0
    d1n = np.linalg.norm(d1)
    if d1n < 1e-10:
        ax2 = (ax + 1) % 3
        e = np.zeros(3); e[ax2] = 1.0
        d1 = e - np.dot(e, t0) * t0
        d1n = np.linalg.norm(d1)
    d1 /= d1n
    d2 = np.cross(t0, d1); d2 /= np.linalg.norm(d2)
    F[0] = [d1, d2, t0]

    for i in range(1, n):
        tp, tc = F[i - 1, 2], tangents[i]
        v = tc - tp
        vsq = float(np.dot(v, v))
        if vsq < 1e-20:
            F[i] = F[i - 1].copy(); F[i, 2] = tc
        else:
            d1p, d2p = F[i - 1, 0], F[i - 1, 1]
            d1c = d1p - (2.0 / vsq) * np.dot(v, d1p) * v
            d2c = d2p - (2.0 / vsq) * np.dot(v, d2p) * v
            d1c = d1c - np.dot(d1c, tc) * tc
            n1 = np.linalg.norm(d1c)
            if n1 < 1e-10:
                ax = int(np.argmin(np.abs(tc)))
                e = np.zeros(3); e[ax] = 1.0
                d1c = e - np.dot(e, tc) * tc
                n1 = np.linalg.norm(d1c)
            d1c /= n1
            d2c = np.cross(tc, d1c); d2c /= np.linalg.norm(d2c)
            F[i] = [d1c, d2c, tc]

    return F


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shape_to_wavelength_shifts(
    shape_points: np.ndarray,
    cal_matrices: dict,
    sensor_locs_from_tip,
    num_chs: int,
    insertion_depth: float = None,
    noise_std: float = 0.0,
) -> dict:
    """Compute per-channel FBG wavelength shifts from a 3-D needle shape.

    Parameters
    ----------
    shape_points : (N, 3) ndarray
        Shape polyline in mm, from entry point (index 0) to needle tip.
    cal_matrices : dict
        ``{float(location_mm_from_tip): ndarray (2, num_chs)}`` – calibration
        matrices keyed by sensor location **from the tip** (mm).
        See ``HyperionDemo._get_cal_matrices_from_needle`` for how to obtain
        this from ``fbgneedle.cal_matrices`` (which is keyed from the base).
    sensor_locs_from_tip : list of float
        Active-area locations from the needle tip (mm), in AA index order.
    num_chs : int
    insertion_depth : float, optional
        Current insertion depth (mm).  Defaults to the polyline's arc length.
    noise_std : float, optional
        Optional Gaussian noise σ (nm) added to each Δλ.

    Returns
    -------
    dict  {ch_num (1-based int): ndarray (num_aas,)}
        Wavelength shifts in nm, ready to be added to reference wavelengths
        to obtain absolute simulated peaks.
    """
    arc = _arc_lengths(shape_points)
    if insertion_depth is None:
        insertion_depth = float(arc[-1])

    # Sensor arc-length positions from the base of the inserted segment
    sensor_s = [max(0.0, insertion_depth - loc) for loc in sensor_locs_from_tip]

    tans = _tangents(shape_points, arc)
    frames = _parallel_transport_frames(tans)

    # Curvature components κx, κy in the material frame
    dtds = _deriv_cols(tans, arc)
    kx = np.einsum('ni,ni->n', dtds, frames[:, 0])
    ky = np.einsum('ni,ni->n', dtds, frames[:, 1])

    # Interpolate at sensor locations and map to body-frame angular velocity:
    #   ω[0] = −(dT/ds)·d2 = −κy   (slot 0 of current_curvatures)
    #   ω[1] = +(dT/ds)·d1 = +κx   (slot 1 of current_curvatures)
    s_min, s_max = arc[0], arc[-1]
    kappa = np.zeros((len(sensor_locs_from_tip), 2))
    for i, s in enumerate(sensor_s):
        sc = float(np.clip(s, s_min, s_max))
        kappa[i, 0] = -float(np.interp(sc, arc, ky))
        kappa[i, 1] = float(np.interp(sc, arc, kx))

    # Inverse calibration: Δλ = pinv(C) @ [ω[0], ω[1]]
    # Use nearest-key lookup so that minor floating-point differences between
    # cal_matrices keys and sensor_location_tip values do not raise a KeyError.
    _cal_keys = np.array(sorted(cal_matrices.keys()), dtype=float)

    def _nearest_cal(loc: float) -> np.ndarray:
        idx = np.argmin(np.abs(_cal_keys - loc))
        return cal_matrices[_cal_keys[idx]]

    num_aas = len(sensor_locs_from_tip)
    delta_lambda = {ch: np.zeros(num_aas) for ch in range(1, num_chs + 1)}
    for aa_idx, loc in enumerate(sensor_locs_from_tip):
        C = _nearest_cal(float(loc))       # (2, num_chs)
        dl = np.linalg.pinv(C) @ kappa[aa_idx]   # (num_chs,)
        for ci in range(num_chs):
            delta_lambda[ci + 1][aa_idx] = float(dl[ci])

    if noise_std > 0.0:
        rng = np.random.default_rng()
        for ch in delta_lambda:
            delta_lambda[ch] += rng.normal(0.0, noise_std,
                                           size=delta_lambda[ch].shape)

    return delta_lambda
