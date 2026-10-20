"""共用的操作锁、动作确认与视觉准备回调，以及订阅清理。"""
from contextlib import contextmanager
import threading

CONTROL_LOCK = threading.Lock()
_callback = None
_vision_callback = None


@contextmanager
def operation(confirm, before_vision=None):
    if confirm is None:
        raise ValueError("真实运动需要 confirm 回调")
    if not CONTROL_LOCK.acquire(blocking=False):
        raise RuntimeError("已有 tray 阶段正在运行")
    global _callback, _vision_callback
    _callback, _vision_callback = confirm, before_vision
    try:
        yield
    finally:
        _callback = None
        _vision_callback = None
        CONTROL_LOCK.release()


def input(prompt):
    if _callback is None:
        raise RuntimeError("确认回调未设置")
    if not bool(_callback("confirm", {"prompt": str(prompt)})):
        raise PermissionError("用户拒绝动作确认")
    return ""


def vision_ready(scene):
    if _vision_callback is not None:
        _vision_callback(str(scene))


def close_wrench_subscribers(controller) -> None:
    buffer = getattr(controller, "wrenches", None)
    for subscriber in getattr(buffer, "_subscribers", ()):
        try:
            subscriber.unregister()
        except Exception:
            pass


@contextmanager
def managed_controller(controller, *, force_log=False):
    """统一阶段退出顺序；记录异常后关闭日志、清理运动并注销订阅。"""
    try:
        yield controller
    except BaseException as exc:
        if force_log:
            controller.mark_force_abort(exc)
        raise
    finally:
        try:
            if force_log:
                controller.close_force_log()
        finally:
            try:
                controller.cleanup()
            finally:
                close_wrench_subscribers(controller)
