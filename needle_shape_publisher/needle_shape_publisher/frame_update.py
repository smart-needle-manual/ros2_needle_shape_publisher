import numpy as np


def stage_pose_translation(stage_position: np.ndarray) -> np.ndarray:
    """Return the world translation contributed by the stage pose.

    The stage pose uses ``position.z`` to encode insertion depth along the
    needle axis. The shape returned by ``needle_shape_sensing`` is already
    parameterised along that axis, so only the lateral ``x/y`` offsets should be
    re-applied as a world-frame translation.

    Parameters
    ----------
    stage_position:
        3-vector ``[x, y, z]`` from ``/stage/state/needle_pose`` where ``x`` and
        ``y`` are lateral guide offsets and ``z`` is insertion depth.
    """
    translation = np.asarray(stage_position, dtype=float).copy()
    translation[2] = 0.0
    return translation


def insertion_point_from_stage_pose(
    entry_point: np.ndarray,
    stage_position: np.ndarray,
) -> np.ndarray:
    """Convert a world-frame skin entry point into the needle-frame air-gap.

    ``stage_position.x/y`` represent lateral guide offsets, while
    ``stage_position.z`` tracks insertion depth. Only the lateral offsets are
    removed when expressing the entry point in the local needle frame.

    Parameters
    ----------
    entry_point:
        3-vector ``[x, y, z]`` describing the skin-entry location in the stage
        frame.
    stage_position:
        3-vector ``[x, y, z]`` from ``/stage/state/needle_pose`` where ``z`` is
        insertion depth rather than an additional world-frame translation.
    """
    insertion_point = np.asarray(entry_point, dtype=float).copy()
    insertion_point[:2] -= np.asarray(stage_position, dtype=float)[:2]
    return insertion_point


def transform_shape(
    pmat: np.ndarray,
    Rmat: np.ndarray,
    stage_position: np.ndarray,
    stage_rotation: np.ndarray,
):
    """Apply the stage-frame rigid transform to a needle shape.

    Parameters
    ----------
    pmat:
        ``(N, 3)`` point array returned by ``needle_shape_sensing`` in the local
        guide/needle frame.
    Rmat:
        ``(N, 3, 3)`` orientation array aligned with ``pmat``.
    stage_position:
        3-vector ``[x, y, z]`` from ``/stage/state/needle_pose`` where only the
        lateral ``x/y`` components contribute to translation.
    stage_rotation:
        ``(3, 3)`` rotation matrix from the same pose.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        The transformed point and orientation arrays in the published frame.
    """
    translation = stage_pose_translation(stage_position)

    pmat_tf = None
    Rmat_tf = None
    if pmat is not None:
        pmat_tf = pmat @ stage_rotation.T + translation.reshape(1, -1)

    if Rmat is not None:
        Rmat_tf = np.einsum('jk,ikl->ijl', stage_rotation, Rmat)

    return pmat_tf, Rmat_tf
