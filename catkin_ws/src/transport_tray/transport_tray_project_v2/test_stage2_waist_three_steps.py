#!/usr/bin/env python3
"""腰部单关节三阶段测试；单位rad，--execute才运动。"""
import argparse

WAIST_SPEED = 0.5
WAIST_ACCELERATION = 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if not parser.parse_args().execute:
        print("计划：抬腰q4=0.2 → 转腰q3=1.5 → 下腰q4=0.5；v={}，acc={}；加--execute执行。".format(
            WAIST_SPEED, WAIST_ACCELERATION))
        return

    import rospy
    from naviai_controller import NaviController, RobotModel
    from tray_control.stage2 import turn_waist
    from tray_control.common.runtime import operation
    from terminal import confirm

    rospy.init_node("test_stage2_waist_three_steps", anonymous=True)
    robot = NaviController(model=RobotModel.WA2)
    rospy.sleep(1.)
    with operation(confirm):
        turn_waist(robot, speed=WAIST_SPEED, acceleration=WAIST_ACCELERATION)


if __name__ == "__main__":
    import sys
    if "--execute" in sys.argv:
        from tray_control.observability.timing import run_with_summary
        sys.exit(run_with_summary(main, "Stage2"))
    else:
        sys.exit(main())
