"""shape_to_wavelength.py

Utility functions for computing FBG wavelength shifts (Δλ) from a 3D needle
shape polyline.  The pipeline is:

  shape (Nx3 points, mm)
    → arc-length parametrisation
    → unit-tangent vectors  (Savitzky-Golay local-quadratic fit, or central
                              differences when scipy is unavailable)
    → rotation-minimising parallel-transport frame
    → curvature components (κx, κy) in the material frame
    → interpolation at each active-area (AA) sensor location
    → inverse calibration matrix:  Δλ_k = pinv(C_k) @ [κx_k, κy_k]

The inverse calibration step mirrors the forward mapping used by the
needle-shape-publisher pipeline:
    [κx_k, κy_k] = C_k @ [Δλ_CH1_k, Δλ_CH2_k, …]

The output ``delta_lambda`` dict contains one entry per FBG channel with an
array of Δλ values (one per AA), formatted so it can be published directly on
the ``sensor/processed`` topic as a ``Float64MultiArray`` message.

References
----------
Conceptual baseline: ``piecewise_exp_test.py`` (``piecewise_exp`` branch of
``rban01/needle_shape_sensing_original``) – Savgol-based tangent estimation +
parallel-transport frame + curvature extraction.
"""

import json
import os

import numpy as np

# Optional scipy for Savitzky-Golay filter
try:
    from scipy.signal import savgol_filter as _savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Optional PyYAML for loading .yaml shape files
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Needle parameter loading
# ---------------------------------------------------------------------------

def load_needle_params_json(needle_param_file: str):
    """Load calibration matrices and sensor geometry from a needle param JSON.

    Parameters
    ----------
    needle_param_file : str
        Path to the needle parameter JSON file (e.g. *_Jig-Calibration*.json).

    Returns
    -------
    cal_matrices : dict
        ``{float(location_mm): np.ndarray shape (2, num_chs)}``
        Calibration matrix for each AA location (in mm from the *tip*).
    sensor_locs_from_tip : list of float
        Sensor locations from the needle tip (mm), sorted by AA index.
    num_chs : int
    num_aas : int
    needle_length : float
        Total needle length (mm).
    """
    with open(needle_param_file, 'r') as fh:
        params = json.load(fh)

    cal_matrices = {
        float(k): np.array(v, dtype=float)
        for k, v in params['Calibration Matrices'].items()
    }

    # Sensor Locations are keyed by AA index (1-based string)
    sensor_locs_raw = {
        int(k): float(v) for k, v in params['Sensor Locations'].items()
    }
    sensor_locs_from_tip = [
        sensor_locs_raw[k] for k in sorted(sensor_locs_raw.keys())
    ]

    num_chs = int(params['# channels'])
    num_aas = int(params['# active areas'])
    needle_length = float(params['length'])

    return cal_matrices, sensor_locs_from_tip, num_chs, num_aas, needle_length


# ---------------------------------------------------------------------------
# Shape file loading
# ---------------------------------------------------------------------------

def load_shape_file(shape_file: str) -> np.ndarray:
    """Load a 3D shape polyline from a YAML or JSON file.

    YAML format (preferred)::

        shape:
          - [x0, y0, z0]
          - [x1, y1, z1]
          ...

    JSON format::

        {"shape": [[x0, y0, z0], [x1, y1, z1], ...]}

    The top-level key may be ``shape`` or ``points``.

    Parameters
    ----------
    shape_file : str
        Path to the YAML/JSON file.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array of 3-D points in mm.
    """
    ext = os.path.splitext(shape_file)[1].lower()

    if ext in ('.yaml', '.yml'):
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load .yaml shape files. "
                "Install it with: pip install PyYAML"
            )
        with open(shape_file, 'r') as fh:
            data = _yaml.safe_load(fh)
    else:
        with open(shape_file, 'r') as fh:
            data = json.load(fh)

    key = 'shape' if 'shape' in data else 'points'
    points = np.asarray(data[key], dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Shape file must contain an Nx3 array of points; "
            f"got shape {points.shape}"
        )

    return points


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_arc_lengths(points: np.ndarray) -> np.ndarray:
    """Compute cumulative arc lengths along a polyline.

    Parameters
    ----------
    points : (N, 3) ndarray

    Returns
    -------
    arc_lengths : (N,) ndarray  starting at 0.0
    """
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def _savgol_derivative(data: np.ndarray, arc_lengths: np.ndarray) -> np.ndarray:
    """Compute first derivative using Savitzky-Golay (column-wise).

    Falls back to ``np.gradient`` when scipy is unavailable or N < 5.
    """
    n = data.shape[0]
    if _SCIPY_AVAILABLE and n >= 5:
        window = min(5, n)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            window = 3
        delta = float(np.mean(np.diff(arc_lengths))) if n > 1 else 1.0
        return np.column_stack([
            _savgol_filter(data[:, i], window_length=window,
                           polyorder=2, deriv=1, delta=delta)
            for i in range(data.shape[1])
        ])
    # Fallback: central differences via np.gradient
    return np.gradient(data, arc_lengths, axis=0)


def compute_tangents(points: np.ndarray, arc_lengths: np.ndarray) -> np.ndarray:
    """Compute unit tangent vectors at each point along the polyline.

    Uses a Savitzky-Golay local quadratic fit (when scipy is available, N≥5)
    for smooth tangent estimation, otherwise falls back to central differences.

    Parameters
    ----------
    points : (N, 3) ndarray
    arc_lengths : (N,) ndarray

    Returns
    -------
    tangents : (N, 3) ndarray  (unit vectors)
    """
    raw = _savgol_derivative(points, arc_lengths)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return raw / norms


def build_parallel_transport_frame(tangents: np.ndarray) -> np.ndarray:
    """Build rotation-minimising (parallel-transport) frames along the curve.

    Each frame is represented as a 3×3 matrix whose rows are
    ``[d1, d2, d3]`` where ``d3 = tangent`` and ``d1``, ``d2`` span the
    normal plane.

    The initial ``d1`` is chosen as the global axis *least aligned* with
    ``tangents[0]``; subsequent frames are obtained by double-reflection
    (rotation-minimising frame propagation).

    Parameters
    ----------
    tangents : (N, 3) ndarray  unit tangent vectors

    Returns
    -------
    frames : (N, 3, 3) ndarray
    """
    n = len(tangents)
    frames = np.zeros((n, 3, 3))

    # ---- initialise first frame ----
    t0 = tangents[0]
    min_axis = int(np.argmin(np.abs(t0)))
    e = np.zeros(3)
    e[min_axis] = 1.0
    d1 = e - np.dot(e, t0) * t0
    d1_norm = np.linalg.norm(d1)
    if d1_norm < 1e-10:
        # Degenerate case – choose another axis
        alt_axis = (min_axis + 1) % 3
        e = np.zeros(3)
        e[alt_axis] = 1.0
        d1 = e - np.dot(e, t0) * t0
        d1_norm = np.linalg.norm(d1)
    d1 = d1 / d1_norm
    d2 = np.cross(t0, d1)
    d2 = d2 / np.linalg.norm(d2)

    frames[0, 0] = d1
    frames[0, 1] = d2
    frames[0, 2] = t0

    # ---- propagate via double-reflection ----
    for i in range(1, n):
        t_prev = frames[i - 1, 2]
        t_curr = tangents[i]

        v = t_curr - t_prev
        v_sq = float(np.dot(v, v))

        if v_sq < 1e-20:
            # Tangents nearly identical – copy frame, update d3
            frames[i] = frames[i - 1].copy()
            frames[i, 2] = t_curr
        else:
            d1_prev = frames[i - 1, 0]
            d2_prev = frames[i - 1, 1]

            # Reflect each material vector through the bisector plane
            d1_curr = d1_prev - (2.0 / v_sq) * np.dot(v, d1_prev) * v
            d2_curr = d2_prev - (2.0 / v_sq) * np.dot(v, d2_prev) * v

            # Re-orthogonalise against t_curr (numerics safety)
            d1_curr = d1_curr - np.dot(d1_curr, t_curr) * t_curr
            n1 = np.linalg.norm(d1_curr)
            if n1 < 1e-10:
                # Fallback – recompute from scratch
                min_ax = int(np.argmin(np.abs(t_curr)))
                e = np.zeros(3)
                e[min_ax] = 1.0
                d1_curr = e - np.dot(e, t_curr) * t_curr
                n1 = np.linalg.norm(d1_curr)
            d1_curr /= n1

            d2_curr = np.cross(t_curr, d1_curr)
            d2_curr /= np.linalg.norm(d2_curr)

            frames[i, 0] = d1_curr
            frames[i, 1] = d2_curr
            frames[i, 2] = t_curr

    return frames


def compute_curvature_components(
    arc_lengths: np.ndarray,
    tangents: np.ndarray,
    frames: np.ndarray,
):
    """Compute material-frame curvature components (κx, κy) along the curve.

    κx = (dt/ds) · d1,   κy = (dt/ds) · d2

    Parameters
    ----------
    arc_lengths : (N,) ndarray
    tangents : (N, 3) ndarray  unit tangent vectors
    frames : (N, 3, 3) ndarray  parallel-transport frames

    Returns
    -------
    kx : (N,) ndarray
    ky : (N,) ndarray
    """
    dtds = _savgol_derivative(tangents, arc_lengths)
    kx = np.einsum('ni,ni->n', dtds, frames[:, 0])
    ky = np.einsum('ni,ni->n', dtds, frames[:, 1])
    return kx, ky


def interpolate_curvature_at_sensors(
    arc_lengths: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    sensor_locs_from_base,
) -> np.ndarray:
    """Interpolate curvature at the arc-length positions of the sensors.

    Parameters
    ----------
    arc_lengths : (N,) ndarray
    kx, ky : (N,) ndarray
    sensor_locs_from_base : sequence of float
        Arc-length position of each sensor measured from the needle base (mm).
        Sensors outside the polyline range are clamped to the nearest endpoint.

    Returns
    -------
    kappa : (num_aas, 2) ndarray  – columns are [κx, κy]
    """
    kappa = np.zeros((len(sensor_locs_from_base), 2))
    s_min, s_max = arc_lengths[0], arc_lengths[-1]
    for i, s in enumerate(sensor_locs_from_base):
        s_clamped = float(np.clip(s, s_min, s_max))
        kappa[i, 0] = float(np.interp(s_clamped, arc_lengths, kx))
        kappa[i, 1] = float(np.interp(s_clamped, arc_lengths, ky))
    return kappa


def curvatures_to_wavelength_shifts(
    kappa_at_sensors: np.ndarray,
    cal_matrices: dict,
    sensor_locs_from_tip,
    num_chs: int,
) -> dict:
    """Convert per-sensor curvatures to FBG wavelength shifts via pseudo-inverse.

    The forward model is  [κx, κy] = C @ [Δλ_1, …, Δλ_num_chs]  (C is 2×num_chs).
    The inverse is        [Δλ_1, …, Δλ_num_chs] = pinv(C) @ [κx, κy].

    Parameters
    ----------
    kappa_at_sensors : (num_aas, 2) ndarray
    cal_matrices : dict
        ``{float(loc_from_tip): ndarray (2, num_chs)}``
    sensor_locs_from_tip : sequence of float  (same order as kappa_at_sensors rows)
    num_chs : int

    Returns
    -------
    delta_lambda : dict
        ``{ch_num (1-based int): ndarray (num_aas,)}``
    """
    num_aas = len(sensor_locs_from_tip)
    delta_lambda = {ch: np.zeros(num_aas) for ch in range(1, num_chs + 1)}

    for aa_idx, loc in enumerate(sensor_locs_from_tip):
        kappa = kappa_at_sensors[aa_idx]          # shape (2,)
        C = cal_matrices[float(loc)]               # shape (2, num_chs)
        C_pinv = np.linalg.pinv(C)                 # shape (num_chs, 2)
        delta_lam = C_pinv @ kappa                 # shape (num_chs,)
        for ch_idx in range(num_chs):
            delta_lambda[ch_idx + 1][aa_idx] = float(delta_lam[ch_idx])

    return delta_lambda


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def shape_to_wavelength_shifts(
    shape_points: np.ndarray,
    needle_param_file: str,
    insertion_depth: float = None,
    noise_std: float = 0.0,
) -> dict:
    """Compute FBG wavelength shifts (Δλ) from a 3D needle shape polyline.

    ``shape_points`` represents the full inserted portion of the needle
    (from the tissue entry point at index 0 to the needle tip at index -1).
    Sensor locations outside the polyline (i.e. proximal sensors not yet
    inserted) are clamped to the nearest polyline endpoint.

    Parameters
    ----------
    shape_points : (N, 3) ndarray
        3-D polyline in mm.
    needle_param_file : str
        Path to the needle parameter JSON file containing ``'Calibration
        Matrices'`` and ``'Sensor Locations'``.
    insertion_depth : float, optional
        Current insertion depth (mm).  Defaults to the total arc length of
        ``shape_points``.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to each Δλ (nm).

    Returns
    -------
    delta_lambda : dict
        ``{ch_num (1-based int): ndarray (num_aas,)}``
        Wavelength shifts per channel and active area (nm).
    """
    (cal_matrices, sensor_locs_from_tip,
     num_chs, num_aas, _) = load_needle_params_json(needle_param_file)

    arc_lengths = compute_arc_lengths(shape_points)
    total_length = float(arc_lengths[-1])

    if insertion_depth is None:
        insertion_depth = total_length

    # Sensor positions measured from the base of the inserted segment
    # sensor_loc_from_tip  →  sensor_loc_from_base = insertion_depth - loc_from_tip
    sensor_locs_from_base = [
        float(max(0.0, insertion_depth - loc))
        for loc in sensor_locs_from_tip
    ]

    tangents = compute_tangents(shape_points, arc_lengths)
    frames = build_parallel_transport_frame(tangents)
    kx, ky = compute_curvature_components(arc_lengths, tangents, frames)
    kappa_at_sensors = interpolate_curvature_at_sensors(
        arc_lengths, kx, ky, sensor_locs_from_base
    )
    delta_lambda = curvatures_to_wavelength_shifts(
        kappa_at_sensors, cal_matrices, sensor_locs_from_tip, num_chs
    )

    if noise_std > 0.0:
        rng = np.random.default_rng()
        for ch in delta_lambda:
            delta_lambda[ch] = delta_lambda[ch] + rng.normal(
                0.0, noise_std, size=delta_lambda[ch].shape
            )

    return delta_lambda
