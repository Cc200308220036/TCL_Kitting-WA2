"""BASE下XYZ及xyz欧拉角的终端结果输出。"""
from scipy.spatial.transform import Rotation
from ..common.state import validate_pose7


def pose6d(pose):
    values = validate_pose7(pose)
    rpy = Rotation.from_quat(values[3:]).as_euler("xyz", degrees=True)
    return [*values[:3], *rpy.tolist()]


def print_pose6d(label, pose):
    x, y, z, roll, pitch, yaw = pose6d(pose)
    print("{}：X={:+.6f}m  Y={:+.6f}m  Z={:+.6f}m  Roll={:+.6f}°  Pitch={:+.6f}°  Yaw={:+.6f}°".format(
        label, x, y, z, roll, pitch, yaw), flush=True)


def print_place_result(expected, actual, release_reason=None, radius_limit=False):
    print_pose6d("P_place目标6D位姿（BASE）", expected)
    print_pose6d("实际放置顶层tray的视觉6D位姿（BASE）", actual)
    delta_mm = 1000 * (actual[2] - expected[2])
    print("实际tray Z − P_place Z = {:+.3f}mm（正值高于目标，负值低于目标）".format(delta_mm), flush=True)
    if release_reason or radius_limit:
        print("释放路径：未独立确认入槽；{}".format(release_reason or "螺旋半径到限后释放"), flush=True)
    return delta_mm
