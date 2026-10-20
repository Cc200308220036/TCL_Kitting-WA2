import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))
from ultralytics import YOLO


def seg_flange(rgb_img):
    rgb_seg_mask = np.zeros([rgb_img.shape[0], rgb_img.shape[1]], dtype=np.uint8)
    rgb_seg_result = model.predict(rgb_img, save=True)  # predict on an image
    result = rgb_seg_result[0].to('cpu')
    det_res = result.boxes.data.numpy()
    box_len = det_res.shape[0]
    if box_len == 0:
        seg = np.zeros_like(rgb_img, dtype=np.uint8)  # 未检测到任何目标，就赋全黑图
    else:
        for k in range(box_len):  # 把检测到的结果全都扣进来
            mask_ = result.masks[k].data.numpy()[0].astype(np.uint8)
            rgb_seg_mask[mask_ == 1] = 1
        seg_res_img = cv2.add(rgb_img, np.zeros(np.shape(rgb_img), dtype=np.uint8), mask=rgb_seg_mask)
        seg = seg_res_img

    seg = cv2.addWeighted(rgb_img, 0.5, seg, 0.5, 0)

    cv2.imshow('seg', seg)
    cv2.waitKey(0)


if __name__ == '__main__':
    model = YOLO('/home/zhurui/work/source/yolo11_seg/runs/train/exp/weights/best.pt')
    img_path = '/media/zhurui/DATA/workspace/data/seg_task/images/1729596746000_3.png'
    image = cv2.imread(img_path)
    seg_flange(image)
