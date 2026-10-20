#!/usr/bin/env python3
"""Minimal action call: send labels, wait, then consume cached RGB-D/masks."""

import rospy
from naviai_yolo.ros.seg_action_client import SegActionClient


def main():
    rospy.init_node('seg_action_call_example')
    client = SegActionClient(timeout_sec=15.0)
    labels = rospy.get_param('~labels', ['wp_2'])
    if not client.send_goal(labels):
        rospy.logwarn('Segmentation failed, was cancelled or timed out')
        return

    if not client.last_items:
        rospy.logwarn('Action succeeded but detection feedback was not received')
        return

    # RGB-D images and masks are ROS Image messages.
    color_image = client.last_color_image
    depth_image = client.last_depth_image
    rospy.loginfo(f'RGB: {color_image.width}x{color_image.height}, depth: {depth_image.encoding}')
    for item in client.last_items:
        rospy.loginfo(f'{item.label}: confidence={item.confidence:.3f}, bbox={list(item.bbox)}')
        # Use CvBridge().imgmsg_to_cv2(item.mask, 'mono16') to obtain a NumPy mask.


if __name__ == '__main__':
    main()
