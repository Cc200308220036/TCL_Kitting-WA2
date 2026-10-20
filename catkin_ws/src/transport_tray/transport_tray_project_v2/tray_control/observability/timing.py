"""控制周期诊断与状态耗时统计；仅观测，不查询或控制机器人。"""
from contextlib import contextmanager
import threading
import time

_active = None


class CycleTiming:
    def __init__(self, period):
        self.period = period
        self.previous = None

    def begin(self, now):
        self.started = now
        self.values = dict(tcp=0.0, wrench=0.0, servo=0.0, log=0.0)

    @contextmanager
    def section(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.values[name] += time.monotonic() - started

    def finish(self, rate):
        started = time.monotonic()
        self.values["active"] = started - self.started
        self.values["compute"] = max(
            0.0,
            self.values["active"]
            - sum(self.values[name] for name in ("tcp", "wrench", "servo", "log")),
        )
        rate.sleep()
        self.values["sleep"] = time.monotonic() - started
        self.previous = self.values

    def detail(self, gap: float) -> str:
        """解释上一轮控制循环耗时；rate.sleep 的正常周期等待单独列出。"""
        timing = self.previous
        if timing is None:
            return "上一轮分段计时不可用"
        active = timing["active"]
        sleep = timing["sleep"]
        planned_sleep = max(0.0, self.period - active)
        scheduling = max(0.0, gap - active - sleep)
        return (
            "上一轮耗时ms: TCP={tcp:.1f} 腕力={wrench:.1f} "
            "计算/状态切换={compute:.1f} ServoL发布={servo:.1f} "
            "日志={log:.1f} rate.sleep={sleep:.1f}(超计划={sleep_overrun:.1f}) "
            "循环外调度={scheduling:.1f}"
        ).format(
            tcp=1000.0 * timing["tcp"],
            wrench=1000.0 * timing["wrench"],
            compute=1000.0 * timing["compute"],
            servo=1000.0 * timing["servo"],
            log=1000.0 * timing["log"],
            sleep=1000.0 * sleep,
            sleep_overrun=1000.0 * max(0.0, sleep - planned_sleep),
            scheduling=1000.0 * scheduling,
        )

class StateTimer:
    def __init__(self, group, clock=time.monotonic):
        self.clock = clock
        self.current = (group, "初始化/其他")
        self.since = clock()
        self.totals = {}
        self.visits = {}

    def switch(self, group, label):
        key = (group or self.current[0], label)
        if key == self.current:
            return
        self.finish()
        self.current = key

    def finish(self):
        now = self.clock()
        self.totals[self.current] = self.totals.get(self.current, 0.0) + now - self.since
        self.visits[self.current] = self.visits.get(self.current, 0) + 1
        self.since = now

    def report(self, status):
        self.finish()
        print("\n本次执行状态耗时汇总（{}；单位秒，重复状态累计）：".format(status), flush=True)
        for group in dict.fromkeys(key[0] for key in self.totals):
            rows = [(key, seconds) for key, seconds in self.totals.items() if key[0] == group]
            print("{} 总计 {:.3f}s".format(group, sum(seconds for _, seconds in rows)))
            for key, seconds in rows:
                print("  {}：{:.3f}s（{}段）".format(key[1], seconds, self.visits[key]))
        print("本次合计：{:.3f}s；包含初始化、反馈到位、清理及人工等待。".format(sum(self.totals.values())), flush=True)


def state(label, group=None):
    if _active is not None and threading.current_thread() is threading.main_thread():
        _active.switch(group, label)


@contextmanager
def phase(group, label):
    previous = _active.current if _active is not None else None
    state(label, group)
    try:
        yield
    finally:
        if previous is not None:
            state(previous[1], previous[0])


def run_with_summary(main, group):
    """四个独立CLI共享的finally汇总；失败时只统计已执行部分。"""
    global _active
    _active = StateTimer(group)
    status = "异常/中断，已执行部分"
    try:
        result = main()
        status = "完成" if result in (None, 0) else "失败，已执行部分"
        return result
    finally:
        try:
            _active.report(status)
        finally:
            _active = None


def print_task_summary(totals):
    """复用详细计时归并任务阶段，不重复计时或累计并行视觉耗时。"""
    names = {"Stage0": "到预抓取位", "Stage1": "抓取tray（含抬升）", "Stage2": "转腰",
             "Stage3单层": "放置", "Stage3多层": "放置", "返回取料": "返回到初始位置"}
    tasks, other = {}, 0.
    for (group, _), seconds in totals.items():
        layer, _, stage = group.rpartition("/")
        if stage in names:
            label = "{}搬运 / {}".format(layer, names[stage])
            tasks[label] = tasks.get(label, 0.) + seconds
        else:
            other += seconds
    tasks["初始化/阶段衔接"] = other
    print("\n总阶段任务耗时汇总（不含人工确认等待；单位秒）：")
    elapsed, shown_ms = 0., 0
    for label, seconds in tasks.items():
        elapsed += seconds
        next_ms = round(elapsed * 1000)
        print("  {}：{:.3f}s".format(label, (next_ms - shown_ms) / 1000))
        shown_ms = next_ms
    print("总阶段合计：{:.3f}s".format(sum(totals.values())), flush=True)


class LayerStateTimer(StateTimer):
    layer = 0

    def switch(self, group, label):
        group = group or self.current[0]
        if self.layer and not group.startswith("第"):
            group = "第{}层/{}".format(self.layer, group)
        super().switch(group, label)

    def finish(self):
        if self.current[1] in ("人工确认等待", "WAIT_DESCENT_CONFIRM"):
            self.since = self.clock()
        else:
            super().finish()

    def report(self, status):
        self.finish()
        print("\n搬运耗时汇总（{}，不含人工确认等待）：".format(status), flush=True)
        for group in dict.fromkeys(key[0] for key in self.totals):
            rows = [(key[1], seconds) for key, seconds in self.totals.items() if key[0] == group]
            print("{}：{:.3f}s".format(group, sum(seconds for _, seconds in rows)))
            for label, seconds in rows:
                print("  {}：{:.3f}s".format(label, seconds))
        print("总计：{:.3f}s".format(sum(self.totals.values())), flush=True)
        print_task_summary(self.totals)
