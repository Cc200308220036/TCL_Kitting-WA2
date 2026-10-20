"""Paired analysis records and visualizations, independent of ROS and YOLO."""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from uuid import uuid4

import cv2
import numpy as np

from naviai_yolo.configuration import resolve_output_path
from naviai_yolo.postprocess import mask_to_uint8


# OpenCV uses BGR: red, yellow, blue, green, purple, cyan, orange, pink.
LABEL_PALETTE = ((0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 200, 0),
                 (180, 0, 180), (255, 255, 0), (0, 128, 255), (180, 105, 255))
TEXT_FONT_SCALE = 1.0
TEXT_THICKNESS = 2


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def _assign_label_colors(detections, label_colors):
    for detection in detections:
        if detection.label not in label_colors:
            label_colors[detection.label] = LABEL_PALETTE[len(label_colors) % len(LABEL_PALETTE)]


def render_overlay(color, detections, label_colors=None):
    """Draw each label in one color; reuse a mutable label_colors map across frames."""
    label_colors = {} if label_colors is None else label_colors
    _assign_label_colors(detections, label_colors)
    overlay = color.copy()
    for detection in detections:
        foreground = detection.mask > 0
        tint = np.asarray(label_colors[detection.label], dtype=np.float32)
        overlay[foreground] = np.round(
            overlay[foreground] * 0.65 + tint * 0.35).astype(np.uint8)
    for index, detection in enumerate(detections):
        tint = label_colors[detection.label]
        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tint, 2)
        cv2.putText(overlay, f'{index:03d} {detection.label} {detection.confidence:.2f}',
                    (x1, max(y1 - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX,
                    TEXT_FONT_SCALE, tint, TEXT_THICKNESS, cv2.LINE_AA)
    return overlay


def _write_json(path, value):
    # Publish only a closed, complete document; never truncate the previous marker.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.chmod(temporary, 0o644)  # Completed JSON is readable by other users, writable only by its owner.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Failed to write image: {path}')


class DebugWriter:
    def __init__(self, config, server_kind, session, warn):
        self.save_records = config.get('save_records', False)
        self.save_visualizations = config.get('save_visualizations', False)
        for value in (self.save_records, self.save_visualizations):
            if not isinstance(value, bool):
                raise ValueError('debug save_records/save_visualizations must be YAML booleans')
        self.warn = warn
        self.run_id = None
        self.run_dir = None
        self._sequence = 0
        self._lock = threading.Lock()
        self._label_colors = {}
        if not self.enabled:
            return
        try:
            root = resolve_output_path(config.get('output_dir', 'data/seg'))
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            self.run_id = f'{stamp}_{server_kind}_{uuid4().hex[:8]}'
            self.run_dir = root / self.run_id
            self.run_dir.mkdir(parents=True, exist_ok=False)
            _write_json(self.run_dir / 'session.json', {
                'schema_version': 1, 'run_id': self.run_id,
                'started_at': utc_now(), 'server_kind': server_kind,
                'settings': session,
                'debug': {'output_dir': str(root), 'save_records': self.save_records,
                          'save_visualizations': self.save_visualizations},
            })
        except Exception as exc:
            self.save_records = self.save_visualizations = False
            self.warn(f'Debug recording disabled: {exc}')

    @property
    def enabled(self):
        return self.save_records or self.save_visualizations

    def next_sample_id(self):
        if not self.enabled:
            return None
        with self._lock:
            self._sequence += 1
            return f'{self._sequence:06d}'

    def save(self, sample_id, color, depth, detections, metadata):
        if not self.enabled:
            return
        if self.save_records:
            try:
                self._save_record(sample_id, color, depth, detections, metadata)
            except Exception as exc:
                self.warn(f'Debug record {self.run_id}/{sample_id} failed: {exc}')
        if self.save_visualizations and color is not None:
            try:
                path = self.run_dir / 'visualizations' / f'{sample_id}.jpg'
                path.parent.mkdir(exist_ok=True)
                # Allocate colors atomically; image rendering and I/O stay outside the lock.
                with self._lock:
                    _assign_label_colors(detections, self._label_colors)
                    label_colors = dict(self._label_colors)
                overlay = render_overlay(color, detections, label_colors=label_colors)
                if not detections:
                    cv2.putText(overlay, metadata['status'], (8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE,
                                (0, 180, 255), TEXT_THICKNESS, cv2.LINE_AA)
                _write_image(path, overlay)
            except Exception as exc:
                self.warn(f'Debug visualization {self.run_id}/{sample_id} failed: {exc}')

    def _save_record(self, sample_id, color, depth, detections, metadata):
        path = self.run_dir / 'records' / sample_id
        path.mkdir(parents=True, exist_ok=False)
        meta = dict(metadata, schema_version=1, run_id=self.run_id,
                    sample_id=sample_id, saved_at=utc_now(), record_complete=False)
        meta['files'] = {'color': 'color.png' if color is not None else None,
                         'depth': 'depth.npy' if depth is not None else None,
                         'detections': 'detections.json'}
        for name, array in (('color', color), ('depth', depth)):
            meta[name] = dict(meta.get(name, {}),
                              shape=list(array.shape) if array is not None else None,
                              dtype=str(array.dtype) if array is not None else None)
        # Leave an explicit incomplete record if a later write fails.
        _write_json(path / 'meta.json', meta)
        if color is not None:
            _write_image(path / 'color.png', color)
        if depth is not None:
            np.save(str(path / 'depth.npy'), depth, allow_pickle=False)
        entries = []
        if detections:
            (path / 'masks').mkdir()
        for index, detection in enumerate(detections):
            mask_path = f'masks/{index:03d}.png'
            _write_image(path / mask_path, mask_to_uint8(detection.mask))
            entries.append({
                'instance_id': index, 'source_index': int(detection.source_index),
                'label': detection.label, 'confidence': float(detection.confidence),
                'bbox': [int(value) for value in detection.bbox], 'mask': mask_path,
            })
        _write_json(path / 'detections.json', entries)
        meta['record_complete'] = True
        _write_json(path / 'meta.json', meta)
