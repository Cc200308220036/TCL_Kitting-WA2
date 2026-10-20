"""Camera-driven instance segmentation action."""
import time
from uuid import uuid4
import rospy
import actionlib
import numpy as np
import torch

from cv_bridge import CvBridge, CvBridgeError
from navi_types.msg import InstSegAction, InstSegFeedback, InstSegResult, DetItem

from naviai_yolo.configuration import load_config, resolve_package_path, validate_model_config
from naviai_yolo.ros.debug_recording import DebugRecorder
from naviai_yolo.ros.camera_sub import CameraSubscriber
from naviai_yolo.postprocess import mask_to_mono16, postprocess_segmentation
from naviai_yolo.segmentation import NaviaiSeg

# 与 seg_action_client / grasp 等客户端保持一致
ACTION_SERVER_NAME = 'naviai_manip_inst_seg_action'
FEEDBACK_STATUS_OK = 200
LOOP_RATE_HZ = 30
# 单次任务的墙钟期限；同步推理/写盘返回后必须再次检查。
TASK_TIMEOUT_SEC = 10.0


class _ActionStopped(Exception):
    def __init__(self, reason):
        super().__init__(f'Action stopped: {reason}')
        self.reason = reason


class SegActionServer:
    def __init__(self):
        self.bridge = CvBridge()
        model_config = load_config('model.yaml', rospy.get_param('~model_config', ''))
        validate_model_config(model_config)
        seg_model_path = model_config['seg_model_path']
        self.naviai_seg = NaviaiSeg(resolve_package_path(seg_model_path))
        self.conf = model_config['conf']
        self.imgsz = model_config['imgsz']
        self.debug = DebugRecorder(model_config, 'seg_action', self.bridge,
                                   endpoint=ACTION_SERVER_NAME, mode='yolo')
        self._warm_up_yolo()

        self.camera_subscriber = CameraSubscriber()
        self.camera_subscriber.start_subscribing()
        self.server = actionlib.SimpleActionServer(
            ACTION_SERVER_NAME, InstSegAction, self.execute_callback, False)
        self.server.start()
        rospy.loginfo(f"Segmentation Action Server ready: {ACTION_SERVER_NAME}")

    def _warm_up_yolo(self):
        if not torch.cuda.is_available():
            return
        try:
            if isinstance(self.imgsz, list):
                h, w = self.imgsz[0], self.imgsz[1]
            else:
                h = w = self.imgsz
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            self.naviai_seg.predict(dummy, self.conf, self.imgsz)
            rospy.loginfo("YOLO warm-up done.")
        except Exception as e:
            rospy.logwarn(f"YOLO warm-up failed: {e}")

    def _to_det_items(self, detections):
        items = []
        for detection in detections:
            # 反馈仍用 mono16，与现有客户端一致（前景 65535，背景 0）
            item = DetItem()
            item.label = detection.label
            item.confidence = detection.confidence
            item.bbox = list(detection.bbox)
            item.mask = self.bridge.cv2_to_imgmsg(
                mask_to_mono16(detection.mask), encoding='mono16')
            items.append(item)
        return items

    def _stop_reason(self, deadline):
        if self.server.is_preempt_requested():
            return 'cancelled'
        if rospy.is_shutdown():
            return 'shutdown'
        if time.monotonic() >= deadline:
            return 'timeout'
        return None

    def _check_active(self, deadline):
        reason = self._stop_reason(deadline)
        if reason:
            raise _ActionStopped(reason)

    def _wait_active(self, duration, deadline):
        # Short wall-clock waits keep cancellation/timeout checks independent of /clock.
        self._check_active(deadline)
        time.sleep(min(duration, max(0.0, deadline - time.monotonic())))
        self._check_active(deadline)

    def _finish_action(self, result, detected_success, reason=None, deadline=None):
        try:
            self.camera_subscriber.clear_camera_data()
        except Exception as exc:
            rospy.logerr(f'Camera cleanup failed: {exc}')
            reason = reason or f'camera cleanup failed: {exc}'
        # Recheck after cleanup, immediately before committing a terminal state.
        if deadline is not None:
            reason = self._stop_reason(deadline) or reason
        result.success = bool(detected_success and reason is None)
        if reason == 'cancelled':
            rospy.loginfo("Task preempted")
            self.server.set_preempted(result)
        elif result.success:
            rospy.loginfo("Task completed: target detected")
            self.server.set_succeeded(result)
        else:
            rospy.loginfo(f"Task failed: {reason or 'no target detected'}")
            self.server.set_aborted(result)

    def execute_callback(self, goal):
        rospy.loginfo(f"Segmentation request, labels={goal.labels}")
        target_labels = set(goal.labels)

        feedback = InstSegFeedback()
        result = InstSegResult()
        detected_success = False
        deadline = time.monotonic() + TASK_TIMEOUT_SEC
        reason = None
        try:
            if not target_labels:
                rospy.logwarn("Empty target labels, abort action")
                reason = 'empty target labels'
                return
            self.camera_subscriber.clear_camera_data()
            request_id = uuid4().hex
            while True:
                self._wait_active(1.0 / LOOP_RATE_HZ, deadline)

                camera_data = self.camera_subscriber.get_camera_data()
                if camera_data is None:
                    rospy.logdebug_throttle(2.0, "camera data queue empty")
                    continue

                bgr_raw = camera_data.get('rgb',None)
                depth_raw = camera_data.get('depth',None)
                if bgr_raw is None or depth_raw is None:
                    rospy.logwarn_throttle(2.0, "rgb or depth message is None")
                    continue

                try:
                    with self.debug.record(bgr_raw, depth_raw, target_labels=list(goal.labels),
                                           request_id=request_id) as sample:
                        cv_bgr = self.bridge.imgmsg_to_cv2(bgr_raw, 'bgr8')
                        if cv_bgr is None or cv_bgr.size == 0:
                            raise ValueError('Color image is empty')
                        sample['color'] = cv_bgr
                        self._check_active(deadline)
                        with torch.inference_mode():
                            boxes, masks, names = self.naviai_seg.predict(cv_bgr, self.conf, self.imgsz)
                        self._check_active(deadline)
                        detections = postprocess_segmentation(
                            boxes, masks, names, cv_bgr.shape, target_labels=target_labels)
                        feedback.items = self._to_det_items(detections)
                        sample['detections'] = detections
                except CvBridgeError as e:
                    rospy.logerr(f"CV Bridge error: {e}")
                    continue

                # The recorder may have spent time writing files before this point.
                self._check_active(deadline)
                feedback.color_image = bgr_raw
                feedback.depth_image = depth_raw
                feedback.status = FEEDBACK_STATUS_OK
                feedback.have_objs = bool(feedback.items)
                self.server.publish_feedback(feedback)

                if feedback.have_objs:
                    self._wait_active(0.05, deadline)
                    detected_success = True
                    break
        except _ActionStopped as exc:
            reason = exc.reason
        except Exception as exc:
            rospy.logerr(f'Segmentation action failed: {exc}')
            reason = f'action failed: {exc}'
        finally:
            self._finish_action(result, detected_success, reason=reason, deadline=deadline)


def main():
    rospy.init_node('seg_action_server')
    SegActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
