#!/usr/bin/env python3
"""从当前悬空夹持位置执行单层Z导纳、张爪撤离，记录第一层顶面。"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record-top-only", action="store_true", help="仅重新记录第一层顶面并重建状态，不执行运动")
    parser.add_argument("--state-file", help="输出顶面状态文件，默认使用封装配置")
    options = parser.parse_args()
    if not options.execute:
        print("计划：" + ("仅记录第一层顶面并重建状态" if options.record_top_only else
              "Z下降导纳 → 自动张爪 → 下探/外撤/上抬 → 记录第一层并重建状态") + "；加--execute执行。")
        return

    import rospy
    from tray_control.config import TrayConfig
    from tray_control.stage3 import SingleStage3Place
    from tray_control.common.vision import record_first_layer
    from tray_control.common.runtime import operation, managed_controller
    from tray_control.observability.pose_reporting import print_pose6d
    from tray_control.observability.timing import state as timing_state
    from terminal import confirm, before_vision

    rospy.init_node("test_stage3_single_z_place", anonymous=True)
    config = TrayConfig(state_path=options.state_file)
    print("状态文件：", config.state_path, flush=True)
    if not options.record_top_only:
        args = config.stage3_args(False)
        # 仅本次测试关闭找平及50mm预下降，直接运行原Z导纳。
        with operation(confirm, before_vision):
            with managed_controller(SingleStage3Place(args)) as controller:
                controller.run()
    timing_state("RECORD_TOP")
    with operation(confirm, before_vision):
        if not confirm("record_top", {"prompt": "确认现场只有已放置的第一层tray、双臂已撤离，按Enter更新第一层顶面："}):
            raise PermissionError("用户取消首层记录")
        pose = record_first_layer(config)
    print("第一层tray [x,y,z,qx,qy,qz,qw]：", pose)
    print_pose6d("实际放置第一层tray的6D位姿（BASE）", pose)
    print("首层状态已直接更新，旧多层目标及待补记标志已清除，不生成备份。")


if __name__ == "__main__":
    import sys
    if "--execute" in sys.argv:
        from tray_control.observability.timing import run_with_summary
        sys.exit(run_with_summary(main, "Stage3单层"))
    else:
        sys.exit(main())
