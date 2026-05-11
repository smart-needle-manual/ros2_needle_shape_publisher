import numpy as np

from needle_shape_publisher.frame_update import (
    insertion_point_from_stage_pose,
    stage_pose_translation,
    transform_shape,
)


def test_stage_pose_translation_ignores_insertion_depth_axis():
    stage_position = np.array([4.0, -3.0, 125.0])

    translation = stage_pose_translation(stage_position)

    np.testing.assert_allclose(translation, [4.0, -3.0, 0.0])


def test_insertion_point_only_removes_lateral_stage_offset():
    entry_point = np.array([12.0, -5.0, 18.0])
    stage_position = np.array([4.0, -3.0, 125.0])

    insertion_point = insertion_point_from_stage_pose(entry_point, stage_position)

    np.testing.assert_allclose(insertion_point, [8.0, -2.0, 18.0])


def test_transform_shape_preserves_inserted_shape_length():
    pmat = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 10.0],
        [3.0, 0.0, 20.0],
    ])
    Rmat = np.tile(np.eye(3), (pmat.shape[0], 1, 1))
    stage_position = np.array([0.0, 0.0, 100.0])
    stage_rotation = np.eye(3)

    pmat_tf, Rmat_tf = transform_shape(pmat, Rmat, stage_position, stage_rotation)

    np.testing.assert_allclose(pmat_tf, pmat)
    np.testing.assert_allclose(Rmat_tf, Rmat)


def test_transform_shape_applies_lateral_offset_and_rotation():
    pmat = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    stage_position = np.array([2.0, -1.0, 40.0])
    stage_rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    Rmat = np.tile(np.eye(3), (pmat.shape[0], 1, 1))

    pmat_tf, Rmat_tf = transform_shape(pmat, Rmat, stage_position, stage_rotation)

    np.testing.assert_allclose(
        pmat_tf,
        np.array([
            [2.0, -1.0, 0.0],
            [2.0, -1.0, 5.0],
        ]),
    )
    np.testing.assert_allclose(Rmat_tf, np.tile(stage_rotation, (pmat.shape[0], 1, 1)))
