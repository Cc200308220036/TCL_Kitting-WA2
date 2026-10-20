"""Adapt the bundled YOLO runtime to the project's NumPy output contract."""

from pathlib import Path


class NaviaiSeg:
    def __init__(self, model_path):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f'Segmentation model not found: {model_path}')
        from naviai_yolo.yolo11_seg.ultralytics import YOLO
        self.seg_model = YOLO(str(model_path))

    def predict(self, bgr_image, conf, imgsz):
        # Debug output belongs to the server recorder, never the YOLO runtime.
        result = self.seg_model.predict(
            bgr_image, imgsz=imgsz, conf=conf, save=False,
            save_txt=False, save_crop=False)[0].to('cpu')
        if result.boxes is None or len(result.boxes) == 0:
            return None, None, None
        if result.masks is None:
            raise ValueError('Model returned boxes without segmentation masks; use segmentation weights')
        boxes = result.boxes.data.numpy()
        masks = result.masks.data.numpy()
        return boxes, masks, self.seg_model.names
