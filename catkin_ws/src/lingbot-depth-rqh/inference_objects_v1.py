from cgi import print_directory
import torch
import cv2
import numpy as np
from mdm.model.v2 import MDMModel
import os
import time
from pathlib import Path
# import trimesh

def resolve_model_path(model_ref: str) -> str:
    """Prefer local file path; fallback to Hugging Face repo id."""
    candidate = Path(model_ref)
    if candidate.exists():
        return str(candidate)

    # Make relative paths stable even when running script from other cwd.
    script_dir_candidate = Path(__file__).resolve().parent / model_ref
    if script_dir_candidate.exists():
        return str(script_dir_candidate)

    return model_ref


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ref = os.getenv("LINGBOT_MODEL_PATH", "robbyant/lingbot-depth-pretrain-vitl-14-v0.5/model.pt")
model = MDMModel.from_pretrained(resolve_model_path(model_ref)).to(device)
# model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5').to(device)
# model = model.cuda().float()

data_path =resolve_model_path('examples_objects/grey_box-6')
# data_path = 'evian_natural_mineral_water'
for i in range(3):
    for scene in os.listdir(os.path.join(data_path, 'rgb')):
        print("scene:", scene)
        # Load and prepare inputs
        image = cv2.cvtColor(cv2.imread(os.path.join(data_path, 'rgb', scene)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # image = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        image = torch.from_numpy(image).to(device=device, dtype=torch.float32) / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()

        depth = cv2.imread(os.path.join(data_path, 'depth', scene), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # depth = torch.tensor(depth, dtype=torch.float32, device=device)[None]
        depth = torch.from_numpy(depth).to(device=device, dtype=torch.float32).unsqueeze(0)

        intrinsics = np.loadtxt(os.path.join(data_path, 'cam_K.txt'))
        intrinsics[0] /= w  # Normalize fx and cx by width
        intrinsics[1] /= h  # Normalize fy and cy by height
        # intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]
        intrinsics = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32).unsqueeze(0)

        image = torch.nn.functional.interpolate(
            image, size=(480, 640), mode="bilinear", align_corners=False
        )
        depth = torch.nn.functional.interpolate(
            depth[:, None], size=(480, 640), mode="nearest"
        )[:, 0]

        torch.cuda.synchronize()
        start = time.time()
        # Run inference
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model.infer(
                image,
                depth_in=depth,
                intrinsics=intrinsics
            )
        torch.cuda.synchronize()
        print("infer real time:", time.time() - start)

        depth_pred = output['depth']  # Refined depth map
        points = output['points']      # 3D point cloud

        t_cpu = time.time()
        depth_np = depth_pred.squeeze().detach().cpu().numpy()
        # 清理异常值
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        # 保存真实深度
        print("gpu to cpu time:", time.time() - t_cpu)


        cv2.imwrite(os.path.join(data_path, "depth", scene[:-4]+"_pred.png"), (depth_np*1000).astype(np.uint16))
        
        # 深度图可视化
        max_depth = 0.002
        depth = depth_np / 1000.0
        depth[depth > max_depth] = max_depth
        depth_norm = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(data_path, "depth", scene[:-4]+"_pred_vis.png"), depth_color)
