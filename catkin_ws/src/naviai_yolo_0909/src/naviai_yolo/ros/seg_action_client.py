"""Reusable client for the camera-driven segmentation action."""
import time
from threading import Event
import rospy
import actionlib

from actionlib_msgs.msg import GoalStatus
from navi_types.msg import InstSegAction, InstSegGoal
from naviai_yolo.configuration import validate_timeout

ACTION_SERVER_NAME = 'naviai_manip_inst_seg_action'
DEFAULT_TIMEOUT_SEC = 60.0

class SegActionClient:
    def __init__(self, timeout_sec=DEFAULT_TIMEOUT_SEC, save_vis=False):
        validate_timeout(timeout_sec, 'timeout_sec')
        self.client = actionlib.SimpleActionClient(
            ACTION_SERVER_NAME,
            InstSegAction
        )

        self.timeout_sec = timeout_sec
        if save_vis:
            rospy.logwarn('save_vis is deprecated and ignored; configure debug in the server model.yaml')

        self.last_items = []
        self.last_color_image = None
        self.last_depth_image = None
        self.last_status = None

        rospy.loginfo(f"Waiting for action server '{ACTION_SERVER_NAME}'...")
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError(f"Action server '{ACTION_SERVER_NAME}' not available")

        rospy.loginfo("Action server connected.")

    def _reset_cache(self):
        self.last_items = []
        self.last_color_image = None
        self.last_depth_image = None
        self.last_status = None

    def feedback_callback(self, feedback):
        self.last_status = feedback.status

        rospy.loginfo(
            f"Feedback: have_objs={feedback.have_objs}, "
            f"status={feedback.status}, items={len(feedback.items)}"
        )

        if not feedback.have_objs or len(feedback.items) == 0:
            return

        # 缓存最后一次有效检测结果
        self.last_items = list(feedback.items)
        self.last_color_image = feedback.color_image
        self.last_depth_image = feedback.depth_image

        for i, item in enumerate(feedback.items):
            rospy.loginfo(
                f"  [{i}] label={item.label}, "
                f"conf={item.confidence:.2f}, bbox={list(item.bbox)}"
            )

    def send_goal(self, labels):
        self._reset_cache()
        if isinstance(labels, (str, bytes)):
            raise ValueError('labels must be a sequence of class name strings')
        try:
            labels = [] if labels is None else list(labels)
        except TypeError as exc:
            raise ValueError('labels must be a sequence of class name strings') from exc
        if any(not isinstance(label, str) for label in labels):
            raise ValueError('labels must contain only class name strings')
        if not labels:
            rospy.logerr("Empty labels. Goal will not be sent.")
            return False

        goal = InstSegGoal()
        goal.labels = labels

        rospy.loginfo(f"Sending goal, labels={labels}")
        done = Event()
        self.client.send_goal(goal, feedback_cb=self.feedback_callback,
                              done_cb=lambda state, result: done.set())

        rospy.loginfo("Waiting for action result...")
        finished = self._wait_for_done(done, self.timeout_sec)

        if not finished:
            rospy.logwarn(
                f"Client wait timeout after {self.timeout_sec}s, cancelling goal"
            )

            self.client.cancel_goal()

            cancel_finished = self._wait_for_done(done, 2.0)
            state = self.client.get_state()

            rospy.logwarn(
                f"Cancel requested, cancel_finished={cancel_finished}, "
                f"final_state={state}"
            )

            return False

        result = self.client.get_result()
        state = self.client.get_state()
        state_text = self.client.get_goal_status_text()

        if result is None:
            rospy.logerr(
                f"Action finished but result is None. "
                f"state={state}, text={state_text}"
            )
            return False

        rospy.loginfo(
            f"Action done: state={state}, text='{state_text}', "
            f"result.success={result.success}, cached_items={len(self.last_items)}"
        )

        success = (state == GoalStatus.SUCCEEDED and result.success)

        if success and len(self.last_items) == 0:
            rospy.logwarn(
                "Action succeeded, but no valid feedback items were cached. "
                "Check whether server publishes feedback before set_succeeded()."
            )

        return bool(success and self.last_items)

    @staticmethod
    def _wait_for_done(done, timeout_sec):
        # Event waiting uses wall time, including when /clock is paused or absent.
        deadline = time.monotonic() + timeout_sec
        while not rospy.is_shutdown():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if done.wait(min(0.05, remaining)):
                return time.monotonic() <= deadline
        return False


# Deprecated compatibility alias for existing external Python callers.
InstanceSegmentationClient = SegActionClient


def main():
    rospy.init_node('seg_action_client')

    client = SegActionClient(
        timeout_sec=rospy.get_param('~timeout_sec', DEFAULT_TIMEOUT_SEC)
    )

    target_labels = rospy.get_param('~labels', ['wp_2'])

    t0 = time.time()
    success = client.send_goal(target_labels)

    rospy.loginfo(
        f"Elapsed {time.time() - t0:.2f}s, success={success}, "
        f"num_cached_items={len(client.last_items)}"
    )

    if success:
        rospy.loginfo("Task completed successfully.")

        for i, item in enumerate(client.last_items):
            rospy.loginfo(
                f"Final cached item [{i}]: "
                f"label={item.label}, "
                f"conf={item.confidence:.2f}, "
                f"bbox={list(item.bbox)}"
            )
    else:
        rospy.loginfo("Task failed.")


if __name__ == '__main__':
    main()
