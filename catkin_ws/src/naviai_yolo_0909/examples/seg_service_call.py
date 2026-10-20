#!/usr/bin/env python3
"""Minimal service call: create a client, send RGB-D files, then consume masks.

Run after sourcing the ROS/catkin environment, for example:
    python examples/seg_service_call.py _color_path:=/data/color.png
Optional private parameters: depth_path, labels, service_name, wait_timeout_sec.
"""

import rospy
from cv_bridge import CvBridge, CvBridgeError
from naviai_yolo.ros.seg_service_client import SegServiceClient


def main():
    # An existing business node should initialize rospy only once, not per call.
    rospy.init_node('seg_service_call_example')
    try:
        color_path = rospy.get_param('~color_path')  # Required.
        depth_path = rospy.get_param('~depth_path', None)
        labels = rospy.get_param('~labels', [])  # Empty means all classes.

        # Create once and reuse for subsequent requests; no model is loaded here.
        client = SegServiceClient(
            service_name=rospy.get_param('~service_name', 'instance_segmentation_service'),
            wait_timeout_sec=rospy.get_param('~wait_timeout_sec', 10.0),
        )
        response = client.segment_files(
            color_path=color_path, depth_path=depth_path, labels=labels,
        )
        # For existing arrays, replace the call above with:
        # response = client.segment(color=color_bgr, depth=depth_array, labels=labels)
        # color: BGR uint8; depth: same HxW, uint16 mm / float32 m, or None.

        if not response.have_objs:
            rospy.loginfo('Request succeeded; no matching objects detected')
            return 0

        bridge = CvBridge()
        for item in response.items:
            foreground = bridge.imgmsg_to_cv2(item.mask, 'mono16') > 0
            rospy.loginfo(
                f'{item.label}: confidence={item.confidence:.3f}, bbox={list(item.bbox)}, '
                f'foreground_pixels={int(foreground.sum())}'
            )
            # Add business processing here. Keep input RGB-D in the caller;
            # the response contains have_objs/items, not the original images.
    except (KeyError, ValueError, CvBridgeError, rospy.ROSException, rospy.ServiceException) as exc:
        rospy.logerr(f'Segmentation call failed: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
