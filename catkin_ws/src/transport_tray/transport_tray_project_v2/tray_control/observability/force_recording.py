"""Stage3共用CSV：完整持物基线、相对六维腕力及显式状态事件。

事件行不填充腕力，避免把旧采样伪装为切换瞬间采样。文件使用缓冲写入，
不在实时循环中执行flush/fsync；结束或异常清理时关闭并写出缓冲。
"""
from .timing import state as timing_state

import csv
from datetime import datetime
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo


# 各侧六维腕力统一按 Fx、Fy、Fz、Tx、Ty、Tz 排列，列名只生成一次。
WRENCH_COLUMNS = {
    side: tuple(
        tuple("{}_{}_{}_{}".format(side, key, kind, unit)
              for kind in ("baseline", "raw", "delta"))
        for key, unit in (("fx", "n"), ("fy", "n"), ("fz", "n"),
                          ("tx", "nm"), ("ty", "nm"), ("tz", "nm"))
    )
    for side in ("left", "right")
}


class ForceCsvRecorder:
    def __init__(self, directory, baseline, *, prefix="stage3", extra_columns=(), clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.stage = "BASELINE_READY"
        self.closed = False
        self.sequence = 0
        self.state_since_s = 0.0
        self.prefix = prefix
        self.baseline = {side: tuple(float(v) for v in values) for side, values in baseline.items()}
        if set(self.baseline) != {"left", "right"} or any(len(v) != 6 for v in self.baseline.values()):
            raise ValueError("CSV需要左右腕各六维持物基线")
        self.baseline_row = {
            baseline_key: self.baseline[side][index]
            for side, fields in WRENCH_COLUMNS.items()
            for index, (baseline_key, _, _) in enumerate(fields)
        }
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        self.path = str(directory / "{}_force_{}_{}.csv".format(prefix, stamp, os.getpid()))
        columns = ["row_type", "sequence", "elapsed_s", "unix_time_s", "control_elapsed_s",
                   "stage", "state_from", "state_to", "event", "detail"]
        for side, fields in WRENCH_COLUMNS.items():
            columns.extend(column for field in fields for column in field)
            columns.append(side + "_support_fz_n")
        self.columns = list(dict.fromkeys(columns + list(extra_columns)))
        self.file = open(self.path, "x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        self._emit({"row_type": "baseline", "event": "held_baseline_ready"})

    def _emit(self, values):
        if self.closed:
            return
        row = {"sequence": self.sequence, "elapsed_s": self.clock() - self.started,
               "unix_time_s": time.time(), "stage": self.stage, **values}
        row.update(self.baseline_row)
        self.writer.writerow(row)
        self.sequence += 1
        return row

    def set_state(self, stage):
        if stage == self.stage or self.closed:
            return
        previous, self.stage = self.stage, str(stage)
        timing_state(self.stage)
        row = self._emit({"row_type": "state_transition", "state_from": previous, "state_to": self.stage})
        stamp = datetime.fromtimestamp(row["unix_time_s"], ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print("[Stage3状态切换][{}] 北京时间={} CSV t={:.3f}s  {} → {}（上一状态{:.3f}s）".format(
            self.prefix, stamp, row["elapsed_s"], previous, self.stage,
            row["elapsed_s"] - self.state_since_s), flush=True)
        self.state_since_s = row["elapsed_s"]

    def mark(self, kind, label):
        self._emit({"row_type": "event", "event": kind, "detail": label})

    def writerow(self, values):
        """统一补齐持物基线和六维增量，调用方只需提供原始腕力。"""
        self.set_state(values.get("stage", self.stage))
        row = dict(values)
        row["control_elapsed_s"] = row.pop("elapsed_s", "")
        row["row_type"] = "sample"
        for side, fields in WRENCH_COLUMNS.items():
            for index, (_, raw_key, delta_key) in enumerate(fields):
                if raw_key in row:
                    row[delta_key] = float(row[raw_key]) - self.baseline[side][index]
        self._emit(row)

    def sample(self, raw, support=None):
        row = {"stage": self.stage}
        for side, values in raw.items():
            for index, (_, raw_key, _) in enumerate(WRENCH_COLUMNS[side]):
                row[raw_key] = float(values[index])
            if support is not None and side in support:
                row[side + "_support_fz_n"] = float(support[side])
        self.writerow(row)

    def close(self):
        if not self.closed:
            self._emit({"row_type": "event", "event": "log_closed"})
            self.file.close()
            self.closed = True


# 当前记录器与阶段事件转发；计时仍由timing模块独立处理。
_active = None

def activate(recorder):
    global _active
    _active = recorder

def deactivate(recorder):
    global _active
    if _active is recorder:
        _active = None

def mark(kind, label):
    if _active is not None:
        _active.mark(kind, label)

def state(label):
    timing_state(label)
    if _active is not None:
        _active.set_state(label)
