"""回字形几何和逐角点推进的离线验证。"""
import importlib.util
from pathlib import Path
from unittest import TestCase

import numpy as np

spec = importlib.util.spec_from_file_location(
    "_square_path", Path(__file__).resolve().parents[1] / "tray_control/algorithms/search_path.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SquareSearch = module.SquareSearch


class SquarePathTests(TestCase):
    def test_starts_base_positive_y_and_segments_are_axis_aligned(self):
        path = SquareSearch()
        first, finished = path.target(np.zeros(2))
        np.testing.assert_allclose(first, [0., .002])
        self.assertFalse(finished)
        deltas = np.diff(path.points, axis=0)
        self.assertTrue(np.all(np.minimum(abs(deltas[:, 0]), abs(deltas[:, 1])) < 1e-12))

    def test_all_rings_stay_inside_ten_mm_radius(self):
        path = SquareSearch()
        distances = np.linalg.norm(path.points, axis=1)
        self.assertLessEqual(distances.max(), .010 + 1e-12)
        self.assertAlmostEqual(distances.max(), .010)
        self.assertAlmostEqual(np.max(abs(path.points)), .010 / np.sqrt(2))
        np.testing.assert_allclose(path.points[1::6, 1], [.002, .004, .006, .010 / np.sqrt(2)])
        np.testing.assert_allclose(path.points[1::6], path.points[6::6])
        np.testing.assert_allclose(path.points[1:7],
            [[0., .002], [-.002, .002], [-.002, -.002], [.002, -.002], [.002, .002], [0., .002]])

    def test_waits_for_each_corner_and_finishes_after_last_waypoint(self):
        path = SquareSearch()
        current = np.array([-.004, .002])  # 先对齐寻槽中心
        visited = []
        for _ in range(20000):
            target, done = path.target(current)
            if done:
                break
            delta = target - current
            length = np.linalg.norm(delta)
            step = min(length, .002 * .02)
            previous = current.copy()
            current += delta * (step / length)
            self.assertLessEqual(np.linalg.norm(current - previous), .00004 + 1e-12)
            if np.linalg.norm(current - target) <= 1e-9:
                visited.append(target.copy())
        self.assertTrue(done)
        np.testing.assert_allclose(visited, path.points)
        np.testing.assert_allclose(current, path.points[-1])

    def test_invalid_parameters_rejected(self):
        for radius, step in ((0., .003), (.015, 0.), (float('nan'), .003), (.015, float('inf'))):
            with self.assertRaises(ValueError):
                SquareSearch(radius, step)


class CirclePathTests(TestCase):
    def test_ten_closed_circles_spaced_one_mm(self):
        path = module.CircleSearch()
        self.assertEqual(len(path.points), 1291)
        for i in range(10):
            ring = path.points[1 + i * 129:1 + (i + 1) * 129]
            np.testing.assert_allclose(np.linalg.norm(ring, axis=1), (i + 1) * .001)
            np.testing.assert_allclose(ring[0], ring[-1], atol=1e-12)
        self.assertAlmostEqual(path.length, .01 + 2 * np.pi * .055, delta=.0001)

    def test_budget_covers_center_alignment_and_complete_outer_circle(self):
        from types import SimpleNamespace
        args = SimpleNamespace(search_max_radius=.010, search_xy_speed=.002)
        module.configure_circle_search(args)
        path = module.CircleSearch()
        current = np.array([-.01, .01])
        elapsed = 0.
        visited = []
        while elapsed <= args.search_timeout:
            target, done = path.target(current)
            if done:
                break
            delta = target - current
            distance = np.linalg.norm(delta)
            current += delta * min(1., .002 * .02 / distance)
            elapsed += .02
            if np.linalg.norm(current - target) <= 1e-9:
                visited.append(target.copy())
        self.assertTrue(done)
        np.testing.assert_allclose(visited, path.points, atol=1e-12)
        np.testing.assert_allclose(current, [.01, 0.], atol=1e-12)
        self.assertFalse(hasattr(args, "duration"))
        self.assertEqual(args.search_xy_speed, .002)
