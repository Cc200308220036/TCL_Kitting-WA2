import unittest

import numpy as np

from fdpose.transforms import compute_relative_pose, apply_relative_pose, validate_transform


class RelativePoseTests(unittest.TestCase):
    def test_compute_relation_in_rotated_object_frame(self):
        object_pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2],
                                [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
        target_pose = np.eye(4)
        target_pose[:3, 3] = [1, 4, 3]
        relation = compute_relative_pose(object_pose, target_pose)
        expected = np.array([[0, 1, 0, 2], [-1, 0, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        np.testing.assert_allclose(relation, expected)
        np.testing.assert_allclose(apply_relative_pose(object_pose, relation), target_pose)

        new_object_pose = np.eye(4)
        new_object_pose[:3, 3] = [3, 1, 0]
        expected[:3, 3] = [5, 1, 0]
        np.testing.assert_allclose(apply_relative_pose(new_object_pose, relation), expected)

    def test_invalid_transforms_are_rejected(self):
        nonfinite = np.eye(4)
        nonfinite[0, 3] = np.nan
        bad_last_row = np.eye(4)
        bad_last_row[3, 0] = 1
        for matrix in (np.eye(3), nonfinite, bad_last_row):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    compute_relative_pose(matrix, np.eye(4))
                with self.assertRaises(ValueError):
                    apply_relative_pose(np.eye(4), matrix)

    def test_helpers_do_not_modify_inputs(self):
        first, second = np.eye(4), np.eye(4)
        second[0, 3] = 0.5
        original = second.copy()
        compute_relative_pose(first, second)
        apply_relative_pose(first, second)
        np.testing.assert_array_equal(second, original)
        self.assertEqual(validate_transform("pose", first.tolist()).dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
