"""独立测试终端确认回调。"""
import builtins
import signal
import threading
from tray_control.observability.timing import phase

def confirm(action, summary):
    # Process signal handling belongs to the CLI, never the imported library.
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if main_thread else None
    if main_thread:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with phase(None, "人工确认等待"):
            return builtins.input(summary.get("prompt", action)).strip() == ""
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)

def before_vision(scene):
    if not confirm("before_vision", {"prompt": "确认{}所需图像已准备好后按Enter：".format(scene)}):
        raise PermissionError("用户取消视觉采集")

