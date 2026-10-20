"""Z向移动平衡点导纳，支持共同或独立增量；向下位移和速度为正。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class MovingZParameters:
    target_force: float = 2.0
    mass: float = 1.0
    damping: float = 50.0
    stiffness: float = 150.0
    equilibrium_gain: float = 0.0005
    force_deadband: float = 0.2
    recovery_hysteresis: float = 0.0
    retreat_distance: float = 0.0005
    retreat_trigger_cycles: int = 3
    speed_limit: float = 0.003
    acceleration_limit: float = 0.020
    travel_limit: Optional[float] = 0.05  # None表示不限制累计位移
    equilibrium_limit: Optional[float] = 0.05  # None表示不限制平衡点

    def __post_init__(self) -> None:
        if (
            isinstance(self.retreat_trigger_cycles, bool)
            or not isinstance(self.retreat_trigger_cycles, int)
            or self.retreat_trigger_cycles < 1
        ):
            raise ValueError("退让触发周期数必须为正整数")
        for name, value in vars(self).items():
            if name in ("travel_limit", "equilibrium_limit") and value is None:
                continue
            if not math.isfinite(value) or (
                value < 0
                if name in ("force_deadband", "recovery_hysteresis")
                else value <= 0
            ):
                raise ValueError(
                    "{}必须有限，死区和恢复迟滞允许为0，其余参数须为正数".format(name)
                )
        if self.travel_limit is not None and self.retreat_distance > self.travel_limit:
            raise ValueError("退让距离不得超过Z行程限幅")


@dataclass
class MovingZAxis:
    """每侧独立积分；外部应在进入控制时以当前TCP作为零位。"""

    displacement: float = 0.0
    velocity: float = 0.0
    equilibrium: float = 0.0
    retreat_active: bool = False
    overload_cycles: int = 0
    retreat_armed: bool = True
    retreat_start: float = 0.0
    retreat_travel: float = 0.0
    retreat_elapsed: float = 0.0

    def step(
        self,
        support_force: float,
        dt: float,
        p: MovingZParameters,
        *,
        downward_only: bool = False,
    ) -> float:
        if not math.isfinite(support_force) or not 0 < dt <= 0.1:
            raise ValueError("支撑力或控制周期无效")
        error = p.target_force - support_force
        # 连续死区：区内不追力，区外仅使用超出死区的误差，避免边界跳变。
        error = math.copysign(max(0.0, abs(error) - p.force_deadband), error)
        if downward_only:
            error = max(error, 0.0)
            self.equilibrium = max(self.equilibrium, self.displacement)
        self.equilibrium += p.equilibrium_gain * error * dt
        if p.equilibrium_limit is not None:
            self.equilibrium = max(
                -p.equilibrium_limit, min(p.equilibrium_limit, self.equilibrium)
            )
        acceleration = (
            error
            - p.damping * self.velocity
            - p.stiffness * (self.displacement - self.equilibrium)
        ) / p.mass
        acceleration = max(
            -p.acceleration_limit, min(p.acceleration_limit, acceleration)
        )
        self.velocity = max(
            -p.speed_limit,
            min(
                p.speed_limit,
                self.velocity + acceleration * dt,
            ),
        )
        if downward_only:
            self.velocity = max(self.velocity, 0.0)
        # 临近行程边界提前减速，避免抵达边界时把速度突然清零。
        if p.travel_limit is not None and self.velocity != 0.0:
            remaining = (
                p.travel_limit - self.displacement
                if self.velocity > 0
                else self.displacement + p.travel_limit
            )
            # 留出一个控制周期的刹车裕量，避免离散积分冲过行程边界。
            braking_speed = max(
                0.0,
                math.sqrt(
                    max(
                        0.0,
                        2.0 * p.acceleration_limit * remaining,
                    )
                )
                - p.acceleration_limit * dt,
            )
            self.velocity = math.copysign(
                min(abs(self.velocity), braking_speed), self.velocity
            )
        proposal = self.displacement + self.velocity * dt
        self.displacement = (
            proposal
            if p.travel_limit is None
            else max(-p.travel_limit, min(p.travel_limit, proposal))
        )
        if self.displacement != proposal:
            self.velocity = 0.0
        return self.displacement

    def step_retreat(self, dt: float, p: MovingZParameters) -> float:
        """从触发指令位置出发执行定距梯形/三角速度曲线，到位即交还导纳。"""
        distance = self.retreat_travel
        ramp = min(
            p.speed_limit / p.acceleration_limit,
            math.sqrt(distance / p.acceleration_limit),
        )
        speed = p.acceleration_limit * ramp
        cruise = (
            max(0.0, (distance - p.acceleration_limit * ramp * ramp) / speed)
            if speed
            else 0.0
        )
        duration = 2 * ramp + cruise
        self.retreat_elapsed = min(duration, self.retreat_elapsed + dt)
        t = self.retreat_elapsed
        if t >= duration:
            self.displacement = self.retreat_start - distance
            self.velocity = 0.0
            self.equilibrium = self.displacement
            self.retreat_active = False
        elif t < ramp:
            self.displacement = self.retreat_start - 0.5 * p.acceleration_limit * t * t
            self.velocity = -p.acceleration_limit * t
        elif t < ramp + cruise:
            self.displacement = self.retreat_start - (
                0.5 * speed * ramp + speed * (t - ramp)
            )
            self.velocity = -speed
        else:
            remaining = duration - t
            self.displacement = (
                self.retreat_start
                - distance
                + 0.5 * p.acceleration_limit * remaining * remaining
            )
            self.velocity = -p.acceleration_limit * remaining
        return self.displacement


def step_common_z(axis, forces, dt, params, descent_params=None):
    """连续超力计数后共同定距上退；完成后恢复单向下降导纳。"""
    if not forces or not all(math.isfinite(f) for f in forces) or not 0 < dt <= 0.1:
        raise ValueError("支撑力或控制周期无效")
    peak = max(forces)
    if axis.retreat_active:
        # 已开始的退让不被瞬时卸载打断，也不重复累计超力次数。
        return axis.step_retreat(dt, params)
    if peak <= params.target_force - params.recovery_hysteresis:
        axis.retreat_armed = True
    if peak > params.target_force:
        axis.overload_cycles = axis.overload_cycles + 1 if axis.retreat_armed else 0
        # 前两个超力周期及退让后持续超力时先停止下压。
        axis.velocity = 0.0
        axis.equilibrium = axis.displacement
        if axis.overload_cycles >= params.retreat_trigger_cycles:
            axis.overload_cycles = 0
            axis.retreat_armed = False
            axis.retreat_start = axis.displacement
            axis.retreat_travel = params.retreat_distance
            if params.travel_limit is not None:
                axis.retreat_travel = min(
                    params.retreat_distance,
                    max(0.0, axis.displacement + params.travel_limit),
                )
            axis.retreat_elapsed = 0.0
            axis.retreat_active = True
            return axis.step_retreat(dt, params)
        return axis.displacement
    axis.overload_cycles = 0
    axis.step(
        sum(forces) / len(forces), dt, descent_params or params, downward_only=True
    )
    return axis.displacement
