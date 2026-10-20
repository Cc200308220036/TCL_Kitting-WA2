import time
import torch
import cv2
import os
import numpy as np
import open3d as o3d
from mdm.model.v2 import MDMModel

total_start = time.time()

# =============================
# 模型加载
# =============================
t0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14.pt').to(device)
model.eval()
print("Model load time:", time.time() - t0)

# =============================
# 数据读取
# =============================
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
image_np = cv2.cvtColor(
    cv2.imread('examples_debug/exp_20260313_135120_vio_medium/rgb/0.png'),
    cv2.COLOR_BGR2RGB
)
depth_raw_np = cv2.imread(
    'examples_debug/exp_20260313_135120_vio_medium/depth/0.png',
    cv2.IMREAD_UNCHANGED
).astype(np.float32) / 1000.0
intrinsics_np = np.loadtxt('examples_debug/exp_20260313_135120_vio_medium/cam_K.txt')
print("Data load time:", time.time() - t0)

# =============================
# 数据预处理
# =============================
t0 = time.time()
h, w = image_np.shape[:2]

image = torch.tensor(
    image_np / 255.0,
    dtype=torch.float32,
    device=device
).permute(2, 0, 1)[None]

depth = torch.tensor(
    depth_raw_np,
    dtype=torch.float32,
    device=device
)[None]

intrinsics_np[0] /= w
intrinsics_np[1] /= h

intrinsics = torch.tensor(
    intrinsics_np,
    dtype=torch.float32,
    device=device
)[None]

print("Preprocess time:", time.time() - t0)

# =============================
# 模型推理
# =============================
for i in range(10):
    t0 = time.time()
    output = model.infer(
        image,
        depth_in=depth,
        intrinsics=intrinsics
    )
    depth_pred = output['depth']
    # print(depth_pred[0][0])
    torch.cuda.synchronize() if device.type == "cuda" else None
    print("Inference time:", time.time() - t0)


# # =============================
# # 保存推理后的深度图（保持单位一致）
# # =============================
# depth_pred_np = depth_pred[0].detach().cpu().numpy()  # 单位：米
# # 米 -> 毫米
# depth_pred_mm = depth_pred_np * 1000.0
# # 防止溢出
# depth_pred_mm = np.clip(depth_pred_mm, 0, 65535)
# # 转为 uint16
# depth_pred_uint16 = depth_pred_mm.astype(np.uint16)
# # 保存路径（同目录）
# save_path = os.path.join("examples/exp_20260302_192358_vio_medium/depth/0_pred.png")
# cv2.imwrite(save_path, depth_pred_uint16)

# # =============================
# # 点云生成
# # =============================
# t0 = time.time()

# # 恢复真实内参
# intr = intrinsics[0].detach().cpu().numpy().copy()
# intr[0] *= w
# intr[1] *= h

# fx, fy = intr[0, 0], intr[1, 1]
# cx, cy = intr[0, 2], intr[1, 2]

# H, W = depth_raw_np.shape
# u, v = np.meshgrid(np.arange(W), np.arange(H))
# rgb_np = image[0].permute(1, 2, 0).detach().cpu().numpy()

# # 原始 depth
# z_raw = depth_raw_np
# x_raw = (u - cx) * z_raw / fx
# y_raw = (v - cy) * z_raw / fy

# points_raw = np.stack((x_raw, y_raw, z_raw), axis=-1).reshape(-1, 3)
# colors_raw = rgb_np.reshape(-1, 3)

# mask_raw = z_raw.reshape(-1) > 0
# points_raw = points_raw[mask_raw]
# colors_raw = colors_raw[mask_raw]

# pcd_raw = o3d.geometry.PointCloud()
# pcd_raw.points = o3d.utility.Vector3dVector(points_raw)
# pcd_raw.colors = o3d.utility.Vector3dVector(colors_raw)

# # 预测 depth
# depth_pred_np = depth_pred[0].detach().cpu().numpy()

# z_pred = depth_pred_np
# x_pred = (u - cx) * z_pred / fx
# y_pred = (v - cy) * z_pred / fy

# points_pred = np.stack((x_pred, y_pred, z_pred), axis=-1).reshape(-1, 3)
# colors_pred = rgb_np.reshape(-1, 3)

# mask_pred = z_pred.reshape(-1) > 0
# points_pred = points_pred[mask_pred]
# colors_pred = colors_pred[mask_pred]

# pcd_pred = o3d.geometry.PointCloud()
# pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
# pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

# print("Point cloud generation time:", time.time() - t0)

# print("Total time:", time.time() - total_start)

# # =============================
# # 可视化
# # =============================
# print("Showing RAW depth point cloud")
# o3d.visualization.draw_geometries([pcd_raw])

# print("Showing PREDICTED depth point cloud")
# o3d.visualization.draw_geometries([pcd_pred])
