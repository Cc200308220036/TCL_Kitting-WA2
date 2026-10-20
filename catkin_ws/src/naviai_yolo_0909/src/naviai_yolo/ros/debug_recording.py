"""Collect ROS source metadata without making debug I/O affect segmentation."""

import time
from contextlib import contextmanager

import rospy

from naviai_yolo.configuration import resolve_package_path
from naviai_yolo.debug_writer import DebugWriter, utc_now


def _image_source(message):
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    secs, nsecs = getattr(stamp, 'secs', 0), getattr(stamp, 'nsecs', 0)
    return {'encoding': getattr(message, 'encoding', None),
            'frame_id': getattr(header, 'frame_id', None),
            'stamp': {'secs': secs, 'nsecs': nsecs} if secs or nsecs else None}


class DebugRecorder:
    def __init__(self, config, server_kind, bridge, **source):
        self.bridge = bridge
        settings = dict(source)
        settings.update(model_path=str(resolve_package_path(config['seg_model_path'])),
                        conf=config['conf'], imgsz=config['imgsz'], mask_threshold=0.5)
        self.writer = DebugWriter(config.get('debug', {}), server_kind, settings, rospy.logwarn)
        if self.writer.enabled:
            rospy.loginfo(f'Debug output: {self.writer.run_dir}')

    @contextmanager
    def record(self, color_message, depth_message, target_labels=None, request_id=None):
        sample = {'color': None, 'detections': []}
        if not self.writer.enabled:
            yield sample
            return
        sample_id = self.writer.next_sample_id()
        received_at, started = utc_now(), time.monotonic()
        error = None
        try:
            yield sample
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            try:
                metadata = {
                    'received_at': received_at, 'processed_at': utc_now(),
                    'processing_ms': round((time.monotonic() - started) * 1000, 3),
                    'request_id': request_id, 'target_labels': target_labels,
                    'status': 'error' if error is not None else (
                        'success' if sample['detections'] else 'no_detections'),
                    'error': error, 'detection_count': len(sample['detections']),
                    'color': dict(_image_source(color_message), saved_encoding='bgr8'),
                    'depth': _image_source(depth_message),
                }
                depth = None
                if self.writer.save_records and depth_message is not None:
                    try:
                        depth = sample.get('depth')
                        if depth is None:
                            depth = self.bridge.imgmsg_to_cv2(depth_message, 'passthrough')
                        if depth is None or depth.size == 0:
                            raise ValueError('Depth image is empty')
                    except Exception as exc:
                        depth = None
                        metadata['depth']['error'] = str(exc)
                        rospy.logwarn(f'Debug depth unavailable for {sample_id}: {exc}')
                # These are standard ROS encodings; unknown encodings have no assumed unit.
                metadata['depth']['unit'] = {'16UC1': 'mm', '32FC1': 'm'}.get(
                    metadata['depth']['encoding'])
                self.writer.save(sample_id, sample['color'], depth,
                                 sample['detections'], metadata)
            except Exception as exc:
                rospy.logwarn(f'Debug recording failed for {sample_id}: {exc}')
