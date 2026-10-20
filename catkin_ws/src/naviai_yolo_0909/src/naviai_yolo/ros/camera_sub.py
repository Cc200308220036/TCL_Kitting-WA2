"""Synchronize RGB and depth messages for segmentation."""
import threading
import rospy
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer

from naviai_yolo.configuration import load_config


class CameraSubscriber:
    def __init__(self):
        self.camera_config = load_config('camera.yaml', rospy.get_param('~camera_config', ''))
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self.rgb_sub = None
        self.depth_sub = None
        self.ts = None

    def start_subscribing(self):
        if self.rgb_sub is None and self.depth_sub is None:
            self.rgb_sub = Subscriber(self.camera_config['camera_rgb_topic'], Image)
            self.depth_sub = Subscriber(self.camera_config['camera_depth_topic'], Image)

            # 使用 TimeSynchronizer 同步 RGB 和深度图像
            self.ts = TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)  # 队列大小
            self.ts.registerCallback(self.image_callback)
            rospy.loginfo("Started subscribing to RGB and Depth topics.")
        else:
            rospy.loginfo("Already subscribed to RGB and Depth topics.")

    def stop_subscribing(self):
        # 停止订阅
        if self.rgb_sub and self.depth_sub:
            self.rgb_sub.unregister()
            self.depth_sub.unregister()
            rospy.loginfo("Stopped subscribing to RGB and Depth topics.")
            # 清空订阅器
            self.rgb_sub = None
            self.depth_sub = None
            self.ts = None
            self.clear_camera_data()

    def image_callback(self, rgb_msg, depth_msg):
        frame = {'rgb': rgb_msg, 'depth': depth_msg}
        with self._frame_lock:
            self._latest_frame = frame

    def get_camera_data(self):
        # Consume at most one frame; never wait for a producer to supply data.
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def clear_camera_data(self):
        with self._frame_lock:
            self._latest_frame = None
