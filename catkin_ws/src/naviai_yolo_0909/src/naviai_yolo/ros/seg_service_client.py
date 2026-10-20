"""One-shot RGB-D segmentation client for the label-filtered InstSeg service."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from navi_types.srv import InstSeg
from naviai_yolo.configuration import validate_timeout


SERVICE_NAME = 'instance_segmentation_service'


class SegServiceClient:
    def __init__(self, service_name=SERVICE_NAME, wait_timeout_sec=10.0):
        validate_timeout(wait_timeout_sec, 'wait_timeout_sec')
        self.bridge = CvBridge()
        rospy.wait_for_service(service_name, timeout=wait_timeout_sec)
        self.service_proxy = rospy.ServiceProxy(service_name, InstSeg)
        rospy.loginfo("Instance segmentation service client initialized")

    def segment(self, color, depth=None, labels=None):
        """Segment BGR uint8 arrays; return a server-filtered InstSegResponse.

        Depth is uint16 (millimetres) or float32 (metres), with the same HxW as
        color. Omitted depth becomes a zero uint16 image. None/empty labels keep
        all detections. The response contains have_objs/items, not input images.
        Invalid input and ROS errors propagate to the caller.
        """
        color = np.asarray(color)
        if color.ndim != 3 or color.shape[2] != 3 or color.size == 0 or color.dtype != np.uint8:
            raise ValueError('color must be a nonempty HxWx3 BGR uint8 image')
        if isinstance(labels, str):
            raise ValueError('labels must be a sequence of class names, not a string')
        target_labels = list(labels) if labels is not None else []
        if any(not isinstance(label, str) for label in target_labels):
            raise ValueError('labels must contain only class name strings')

        if depth is None:
            depth = np.zeros(color.shape[:2], dtype=np.uint16)
        else:
            depth = np.asarray(depth)
        if depth.shape != color.shape[:2] or depth.dtype not in (np.uint16, np.float32):
            raise ValueError('depth must match the color image size and use uint16 or float32')

        return self.service_proxy(
            labels=target_labels,
            color_image=self.bridge.cv2_to_imgmsg(np.ascontiguousarray(color), 'bgr8'),
            depth_image=self.bridge.cv2_to_imgmsg(
                np.ascontiguousarray(depth), '16UC1' if depth.dtype == np.uint16 else '32FC1'),
        )

    def segment_files(self, color_path, depth_path=None, labels=None):
        """Read color and depth (image or .npy), then use the same segment() path."""
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f'Unable to read color image: {color_path}')
        depth = None
        if depth_path is not None:
            if Path(depth_path).suffix.lower() == '.npy':
                try:
                    depth = np.load(str(depth_path), allow_pickle=False)
                except (OSError, ValueError, EOFError) as exc:
                    raise ValueError(f'Unable to read depth array: {depth_path}: {exc}') from exc
                if not isinstance(depth, np.ndarray):
                    depth.close()  # Reject an NPZ archive renamed to .npy.
                    raise ValueError(f'Depth file must contain one NumPy array: {depth_path}')
            else:
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f'Unable to read depth image: {depth_path}')
        return self.segment(color, depth, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color', required=True, help='BGR image file')
    parser.add_argument('--depth', help='Aligned uint16/float32 depth image or .npy file')
    parser.add_argument('--labels', nargs='+', help='Keep these classes; omitted means all')
    parser.add_argument('--service', default=SERVICE_NAME, help='InstSeg service name')
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('seg_service_client')
    try:
        client = SegServiceClient(service_name=args.service)
        response = client.segment_files(args.color, args.depth, args.labels)
    except (ValueError, CvBridgeError, rospy.ROSException, rospy.ServiceException) as exc:
        rospy.logerr(f'Segmentation request failed: {exc}')
        return 1

    rospy.loginfo(f'Segmentation: have_objs={response.have_objs}, items={len(response.items)}')
    for item in response.items:
        rospy.loginfo(f'{item.label}: confidence={item.confidence:.3f}, bbox={list(item.bbox)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
