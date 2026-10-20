"""One-shot YOLO segmentation service with server-side label filtering."""

import time
import numpy as np
import rospy
from navi_types.msg import DetItem
from navi_types.srv import InstSeg, InstSegResponse
from cv_bridge import CvBridge

from naviai_yolo.configuration import load_config, resolve_package_path, validate_model_config
from naviai_yolo.ros.debug_recording import DebugRecorder
from naviai_yolo.ros.image_io import decode_color, decode_depth
from naviai_yolo.segmentation import NaviaiSeg
from naviai_yolo.postprocess import mask_to_mono16, postprocess_segmentation


SERVICE_NAME = 'instance_segmentation_service'


class SegServiceServer:
    def __init__(self, ):
        super().__init__()

        self.bridge = CvBridge()
        if rospy.get_param('~mask_file', '') or rospy.get_param('~mask_label', ''):
            raise ValueError('Mask file service mode was removed; use '
                             'naviai_yolo.mask_io.load_mask_file() locally')
        model_config = load_config('model.yaml', rospy.get_param('~model_config', ''))
        validate_model_config(model_config)
        seg_model_path = model_config['seg_model_path']
        self.naviai_seg = NaviaiSeg(resolve_package_path(seg_model_path))
        self.conf = model_config['conf']
        self.imgsz = model_config['imgsz']
        self._warm_up_yolo()
        self.debug = DebugRecorder(
            model_config, 'seg_service', self.bridge, endpoint=SERVICE_NAME, mode='yolo')

        # Register only after the model and recorder are ready.
        self.inst_seg_service = rospy.Service(SERVICE_NAME, InstSeg, self.inst_seg_callback)
        rospy.loginfo("Instance segmentation service initialized")

    def _warm_up_yolo(self):
        import torch

        if not torch.cuda.is_available():
            return
        try:
            if isinstance(self.imgsz, list):
                h, w = int(self.imgsz[0]), int(self.imgsz[1])
            else:
                h = w = int(self.imgsz)
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            self.naviai_seg.predict(dummy, self.conf, self.imgsz)
            rospy.loginfo('YOLO warm-up done.')
        except Exception as exc:
            rospy.logwarn(f'YOLO warm-up failed: {exc}')

    def inst_seg_callback(self, data):
        print(f"receice request...{time.time()}")
        try:
            with self.debug.record(data.color_image, data.depth_image,
                                   target_labels=list(data.labels)) as sample:
                cv_bgr = decode_color(self.bridge, data.color_image)
                sample['color'] = cv_bgr
                sample['depth'] = decode_depth(self.bridge, data.depth_image, cv_bgr.shape)
                boxes, masks, names = self.naviai_seg.predict(cv_bgr, self.conf, self.imgsz)
                detections = postprocess_segmentation(
                    boxes, masks, names, cv_bgr.shape, target_labels=data.labels)
                items_list = []
                for detection in detections:
                    item = DetItem()
                    item.label = detection.label
                    item.confidence = detection.confidence
                    item.bbox = list(detection.bbox)
                    item.mask = self.bridge.cv2_to_imgmsg(
                        mask_to_mono16(detection.mask), encoding='mono16')
                    items_list.append(item)
                sample['detections'] = detections
        except Exception as exc:
            rospy.logerr(f"Segmentation failed: {exc}")
            raise rospy.ServiceException(f'Segmentation failed: {exc}') from exc
        return InstSegResponse(have_objs=bool(items_list), items=items_list)


def main():
    rospy.init_node('seg_service_server')
    seg_service_server = SegServiceServer()
    rospy.spin()


if __name__ == '__main__':
    main()
