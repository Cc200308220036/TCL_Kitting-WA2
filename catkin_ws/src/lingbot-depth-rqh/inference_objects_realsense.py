import torch
import cv2
import numpy as np
from mdm.model.v2 import MDMModel
import os
import time
# import trimesh

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5/model.pt').to(device)

obj_name = "tag_0"
data_path = f'/home/zhangxiaoxiao/zxx/pose/stereo_test/realsense_data/{obj_name}'

os.makedirs(os.path.join(data_path, "depth_lingbot"), exist_ok=True)
os.makedirs(os.path.join(data_path, "depth_lingbot_vis"), exist_ok=True)
for scene in os.listdir(os.path.join(data_path, 'rgb')):
    print("scene:", scene)
    # Load and prepare inputs
    image = cv2.cvtColor(cv2.imread(os.path.join(data_path, 'rgb', scene)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]

    depth = cv2.imread(os.path.join(data_path, 'depth', scene), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth = torch.tensor(depth, dtype=torch.float32, device=device)[None]

    intrinsics = np.loadtxt(os.path.join(data_path, 'cam_K.txt'))
    intrinsics[0] /= w  # Normalize fx and cx by width
    intrinsics[1] /= h  # Normalize fy and cy by height
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]

    start = time.time()
    # Run inference
    output = model.infer(
        image,
        depth_in=depth,
        intrinsics=intrinsics)
    print("infer time:", time.time() - start)

    depth_pred = output['depth']  # Refined depth map
    points = output['points']      # 3D point cloud

    depth_np = depth_pred.squeeze().detach().cpu().numpy()
    # 保存真实深度
    # cv2.imwrite(os.path.join(data_path, "depth", scene[:-4]+"_pred.png"), (depth_np*1000).astype(np.uint16))
    cv2.imwrite(os.path.join(data_path, "depth_lingbot", scene), (depth_np*1000).astype(np.uint16))
    
    # 深度图可视化
    max_depth = 0.002
    depth = depth_np / 1000.0
    depth[depth > max_depth] = max_depth
    depth_norm = cv2.normalize(
        depth, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(data_path, "depth_lingbot_vis", scene), depth_color)
