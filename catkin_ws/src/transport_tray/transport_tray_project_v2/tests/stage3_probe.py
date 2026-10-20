"""用生产方法构建离线阶段探针，不复制控制判据。"""
import ast
from types import MethodType, SimpleNamespace


def search_controller(**values):
    values.setdefault('return_motion', SimpleNamespace(
        commands=None, settled_since=None, z_axis=None, anchor_offsets=None))
    values.setdefault('yaw_search', SimpleNamespace(
        center=None, offset=0., segment=0, started=None,
        capture_since=None, start_z=None, settled_since=None))
    return SimpleNamespace(**values)


def bind_methods(controller, tree, scope, names):
    scope.setdefault('SimpleNamespace', SimpleNamespace)
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(ast.parse('from __future__ import annotations').body + methods, [])
    exec(compile(ast.fix_missing_locations(module), 'stage3_probe', 'exec'), scope)
    for name in names:
        setattr(controller, name, MethodType(scope[name], controller))
