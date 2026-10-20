"""base XY回字形路径：由内向外闭合方环，第一段沿base +Y。"""

import math
import numpy as np

# 搜索时间预算保留20mm入口对齐余量，不限制实际XY平移。
SEARCH_ALIGNMENT_ALLOWANCE_M = 0.02


def _validate_path_parameters(radius, ring_step):
    if not all(math.isfinite(v) and v > 0 for v in (radius, ring_step)):
        raise ValueError("搜索半径和圈间距必须为有限正数")


class SquareSearch:
    def __init__(self, radius=0.010, ring_step=0.002):
        _validate_path_parameters(radius, ring_step)
        half_width = radius / math.sqrt(2)  # 最远角点满足欧氏半径约束
        points = [(0.0, 0.0)]  # 先到寻槽中心，再沿base +Y开始方环
        width = 0.0
        while width < half_width:
            width = min(width + ring_step, half_width)
            points.extend(
                (
                    (0, width),
                    (-width, width),
                    (-width, -width),
                    (width, -width),
                    (width, width),
                    (0, width),
                )
            )
        self._init_points(points)

    def _init_points(self, points):
        """方形、圆形共用路径数组、推进索引和长度初始化。"""
        self.points = np.asarray(points, dtype=float)
        self.index = 0
        self.length = float(
            np.linalg.norm(
                np.diff(np.vstack((np.zeros(2), self.points)), axis=0), axis=1
            ).sum()
        )

    def target(self, current, *, elapsed=0.0, yaw=0.0, paused=False):
        """上一角点的XY指令到位后才前进，不按时间跳过角点。"""
        if paused:
            return np.asarray(current).copy(), False
        while (
            self.index < len(self.points)
            and np.linalg.norm(np.asarray(current) - self.points[self.index]) <= 1e-9
        ):
            self.index += 1
        finished = self.index == len(self.points)
        return self.points[-1 if finished else self.index].copy(), finished


class CircleSearch(SquareSearch):
    """闭合同心圆，沿base +X连接下一圈；指令到位才推进。"""

    def __init__(self, radius=0.010, ring_step=0.001):
        _validate_path_parameters(radius, ring_step)
        points = [(0.0, 0.0)]
        # 每圈128段；10mm圆弧最大弦偏差约3微米。
        for i in range(1, math.ceil(radius / ring_step) + 1):
            r = min(i * ring_step, radius)
            points.append((r, 0.0))
            points.extend(
                (r * math.cos(theta), r * math.sin(theta))
                for theta in np.linspace(0.0, 2 * math.pi, 129)[1:]
            )
        self._init_points(points)


class SpiralSearch:
    """按有效搜索时间推进螺旋；不改变原螺距、角速度及目标Yaw旋转。"""

    def __init__(self, radius, pitch, angular_speed):
        self.radius = radius
        self.pitch = pitch
        self.angular_speed = angular_speed

    def target(self, current, *, elapsed=0.0, yaw=0.0, paused=False):
        theta = self.angular_speed * elapsed
        radius = self.pitch * theta / (2.0 * math.pi)
        local = np.array([radius * math.cos(theta), radius * math.sin(theta)])
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        return rotation @ local, radius >= self.radius


def make_search_path(args):
    if args.search_path == "square":
        return SquareSearch(args.search_max_radius, args.square_ring_step)
    if args.search_path == "circle":
        return CircleSearch(args.search_max_radius, args.circle_ring_step)
    return SpiralSearch(
        args.search_max_radius, args.search_pitch, args.search_angular_speed
    )


def configure_circle_search(args):
    """预算完整路线、入口对齐、逐点推进周期及20s裕量。"""
    args.circle_ring_step = getattr(args, "circle_ring_step", 0.001)
    path = CircleSearch(args.search_max_radius, args.circle_ring_step)
    args.search_timeout = math.ceil(
        (path.length + SEARCH_ALIGNMENT_ALLOWANCE_M) / args.search_xy_speed
        + len(path.points) * 0.02
        + 20.0
    )


def configure_search_path(args, *, radius=None, ring_step=None):
    """读取所选路线配置；闭合路线沿用完整路径的自动时间预算。"""
    profiles = args.search_paths
    if not isinstance(profiles, dict) or set(profiles) != {
        "square",
        "circle",
        "spiral",
    }:
        raise ValueError("search_paths需要square/circle/spiral三个配置")
    for name, profile in profiles.items():
        expected = {"radius", "pitch" if name == "spiral" else "ring_step"}
        if not isinstance(profile, dict) or set(profile) != expected:
            raise ValueError("search_paths." + name + "字段不完整或含未知字段")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in profile.values()
        ):
            raise ValueError("搜索半径、步长和螺距必须为有限正数")
    if args.search_path not in profiles:
        raise ValueError("未知搜索路线")
    selected = profiles[args.search_path]
    args.search_max_radius = selected["radius"] if radius is None else radius
    args.search_pitch = profiles["spiral"]["pitch"]
    for value in (
        args.search_max_radius,
        args.search_xy_speed,
        args.search_timeout,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("搜索半径、速度、行程及超时必须为有限正数")
    if args.search_path == "spiral":
        return
    step = selected["ring_step"] if ring_step is None else ring_step
    if not math.isfinite(step) or step <= 0:
        raise ValueError("搜索步长必须为有限正数")
    if args.search_path == "circle":
        args.circle_ring_step = step
        configure_circle_search(args)
    else:
        args.square_ring_step = step
        path = SquareSearch(args.search_max_radius, step)
        args.search_timeout = max(
            args.search_timeout,
            (path.length + SEARCH_ALIGNMENT_ALLOWANCE_M) / args.search_xy_speed + 10.0,
        )
