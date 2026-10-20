"""V2 启动配置：仅管理参数，不持有 ROS Service、相机或 GPU 资源。"""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml

from fdpose.paths import CAMERA_CONFIG_DIR, CONFIG_DIR, PROJECT_ROOT, WARMUP_DIR


@dataclass(frozen=True)
class ServerConfig:
    """启动时加载一次；ROS 私有参数 > YAML > 下列默认值。"""

    # 记录实际读取的配置文件；由 ~config_path 指定。
    config_path: str = str(CONFIG_DIR / "fdpose_server_v2.yaml")

    # 正式位姿注册的细化迭代次数。
    est_refine_iter: int = 3
    # 是否裁剪 Mask ROI；边界扩展单位为像素。
    use_roi: bool = True
    roi_margin: int = 40
    # Mask 至少需要的前景像素数。
    min_mask_pixels: int = 50

    # Mask 内有效深度的统计范围，单位为米；不在此处修改深度图。
    min_depth_m: float = 0.05
    max_depth_m: float = 3.0
    # 有效深度像素数同时满足数量阈值和比例阈值。
    min_valid_depth_pixels: int = 50
    min_valid_depth_ratio: float = 0.02

    # 保存请求输入/位姿，以及相机系 Mesh 包围框和坐标轴图片。
    save_debug_files: bool = True # 保存原始数据
    save_axis_vis: bool = True # 保存坐标轴可视化图片（方便直观查看结果）
    save_debug_montage: bool = False # 独立保存位姿/模型点云投影/深度三联图
    # 坐标轴长度（米）及坐标轴线宽（像素）。
    axis_vis_scale: float = 0.1
    axis_vis_thickness: int = 3

    # 标定目录包含 intrinsic_data.npz；加载后为绝对路径。
    camera_info_dir: str = str(CAMERA_CONFIG_DIR)
    # 预热在服务注册前执行，目录包含 warmup.yaml 及固定参考图。
    warmup_enabled: bool = False
    warmup_dir: str = str(WARMUP_DIR / "wp_2")

    # 与 V1 分开的 ROS Service 名称。
    pose_service_name: str = "/fdpose_v2/pose_estimation"
    mesh_service_name: str = "/fdpose_v2/load_mesh"

    def validate(self):
        """启动时拒绝错误类型和非法范围，避免静默改变参数含义。"""
        for name in ("use_roi", "save_debug_files", "save_axis_vis", "save_debug_montage", "warmup_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean (use true/false without quotes)")
        for name, minimum in (("est_refine_iter", 1), ("roi_margin", 0),
                              ("min_mask_pixels", 1), ("min_valid_depth_pixels", 1),
                              ("axis_vis_thickness", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("min_depth_m", "max_depth_m", "min_valid_depth_ratio", "axis_vis_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("pose_service_name", "mesh_service_name", "camera_info_dir", "warmup_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError(
                "Invalid depth range: "
                f"min_depth_m={self.min_depth_m}, max_depth_m={self.max_depth_m}"
            )
        if not 0.0 <= self.min_valid_depth_ratio <= 1.0:
            raise ValueError("min_valid_depth_ratio must be in [0, 1]")
        if self.axis_vis_scale <= 0:
            raise ValueError("axis_vis_scale must be positive")


def _resolve_project_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Config paths must be nonempty strings")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_server_config(get_param, logwarn):
    """加载 V2 参数；传入 rospy.get_param / logwarn，便于无 ROS 测试。

    缺失文件告警后使用默认值；未知 YAML 键告警，便于发现拼写错误。
    类型和范围必须符合配置约定，不将字符串 "false" 隐式转成 True。
    所有参数集中在启动时读取，不支持热更新。
    """
    defaults = ServerConfig()
    path = Path(_resolve_project_path(get_param("~config_path", defaults.config_path)))
    if not path.is_file():
        logwarn("FDPose V2 config not found, use defaults: %s", path)
        values = {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"Server config must be a mapping: {path}")
    known_keys = {field.name for field in fields(defaults)} - {"config_path"}
    for key in values.keys() - known_keys:
        logwarn("Unknown FDPose V2 config key: %s", key)

    def setting(name):
        return get_param(f"~{name}", values.get(name, getattr(defaults, name)))

    config = ServerConfig(
        config_path=str(path),
        est_refine_iter=setting("est_refine_iter"),
        use_roi=setting("use_roi"),
        roi_margin=setting("roi_margin"),
        min_mask_pixels=setting("min_mask_pixels"),
        min_depth_m=setting("min_depth_m"),
        max_depth_m=setting("max_depth_m"),
        min_valid_depth_pixels=setting("min_valid_depth_pixels"),
        min_valid_depth_ratio=setting("min_valid_depth_ratio"),
        save_debug_files=setting("save_debug_files"),
        save_axis_vis=setting("save_axis_vis"),
        save_debug_montage=setting("save_debug_montage"),
        axis_vis_scale=setting("axis_vis_scale"),
        axis_vis_thickness=setting("axis_vis_thickness"),
        camera_info_dir=_resolve_project_path(setting("camera_info_dir")),
        warmup_enabled=setting("warmup_enabled"),
        warmup_dir=_resolve_project_path(setting("warmup_dir")),
        pose_service_name=setting("pose_service_name"),
        mesh_service_name=setting("mesh_service_name"),
    )
    config.validate()
    return config
