import torch
import cv2

from moge.model.v2 import MoGeModel 
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()

import matplotlib.pyplot as plt

import glob
import os
from pathlib import Path

import numpy as np
import argparse



def get_module(root, dotted):
    mod = root
    for p in dotted.split('.'):
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod

def bhwc(t):
    # normalize (B,3,H,W) -> (B,H,W,3)
    if t.dim() == 4 and t.shape[1] == 3:
        return t.permute(0, 2, 3, 1)
    return t


def save_affine(_m,_tem,out):
    # just store; DO NOT return
    affine_points["v"] = bhwc(out.detach().cpu())

def save_scale(_m,_tem,out):
    _scale_buf["v"] = out.detach().view(-1,1,1,1).cpu()

def normalize_intrinsics(K_list, widths, heights):
    """K = [[fx, s, cx],[0, fy, cy],[0,0,1]] per frame."""
    norm = []
    for K, W, H in zip(K_list, widths, heights):
        fx, s, cx = K[0][0], K[0][1], K[0][2]
        fy, cy    = K[1][1], K[1][2]
        norm.append([
            fx , fy ,
            cx , cy ,
            s   # optional
        ])
    return np.array(norm)

def robust_stats(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    z = 0.6745 * (x - med) / mad
    cv = np.std(x) / (np.mean(x) + 1e-12)
    return {"median": med, "MAD": mad, "cv": cv, "z": z}

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="~/data/frames/", help="Directory containing input images")
ap.add_argument("--ext", default="jpg", help="Image file extension (e.g., jpg, png)")
ap.add_argument("--save_path", default="~/moge/MoGe/processed_vid3/", help="Directory to save output .npz files")
ap.add_argument("--max_images", type=int, default=-1, help="Maximum number of images to process (-1 for all)")
args = ap.parse_args()

# image_paths = sorted(glob.glob(os.path.join(str(Path("~").expanduser()),"datasets/kitti/dataset/sequences/00/image_0", "*.png")))
print(os.path.join(args.data_dir, "*." + args.ext))
image_paths = sorted(glob.glob(os.path.join(args.data_dir, "*." + args.ext)))
save_path = Path(args.save_path).expanduser()
save_path.mkdir(parents=True, exist_ok=True)

print(f"Found {len(image_paths)} images ")
i = 0
K_list = []
widths = []
heights = []

# --- attach hooks to the exact modules you listed ---
affine_path = "points_head.output_blocks.4"   # Conv2d that outputs 3 channels

affine_points = {}
_scale_buf = {}
scales = []
hm_affine = get_module(model, affine_path).register_forward_hook(save_affine)
hm_scale  = get_module(model, "scale_head.4").register_forward_hook(save_scale)

for image_path in image_paths:

    
    input_image = cv2.imread(image_path) #Images are already in gray scale
    image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    widths.append(input_image.shape[1])
    heights.append(input_image.shape[0])
    # print(input_image.shape)
    input_image = torch.tensor(input_image/255, dtype=torch.float16, device="cuda").permute(2,0,1)

    affine_points.clear()
    
    # with torch.no_grad():
    output = model.infer(input_image)


    if "v" not in affine_points:
        raise RuntimeError("Hook tensors missing; ensure AFFINE_PATH and SCALE_PATH are correct.")
    points_affine = affine_points["v"]  # (H,W,3)

    scales.append(_scale_buf["v"].cpu().squeeze())

    


    # print(points_affine.shape)

    H1, W1 = output['mask'].shape
    pa = points_affine.permute(0,3,1,2)  # (1,3,H,W)
    pa_rs = torch.nn.functional.interpolate(pa , size=(H1, W1), mode="bilinear", align_corners=False)
    points_affine = pa_rs[0].permute(1,2,0).cpu().numpy()  # (H1,W1,3)

    print(points_affine.shape, image_gray.shape)

    Hf,Wf = points_affine.shape[0], points_affine.shape[1]
    Hg,Wg = image_gray.shape[0], image_gray.shape[1]

    if (Hf != Hg) or (Wf != Wg):
        image_gray = cv2.resize(image_gray, (Wf,Hf), interpolation=cv2.INTER_AREA)
    # print(output['intrinsics'])

    K = output['intrinsics'].cpu().numpy()
    point_3d = output['points'].cpu().numpy()
    mask = output['mask'].cpu().numpy()
    normals = output['normal'].cpu().numpy()
    depth = output['depth'].cpu().numpy()

    file_name = os.path.basename(image_path).replace(".jpg", ".npz")
    np.savez_compressed(
        save_path / file_name,
        points=point_3d,
        points_affine=points_affine,
        gray=image_gray,
        mask=mask,
        normals=normals,
        depth=depth,
        K=K
    )

    # K_list.append(K)

    if args.max_images > 0 and i >= args.max_images - 1:
        break
    i += 1
    print(f"Processed {i} images", end="\r")

hm_affine.remove()  # remove hooks when done
hm_scale.remove()

s_all = torch.stack(scales).view(-1)   # (T,)
print("scale median/mean/std:", 
    s_all.median().item(), 
    s_all.mean().item(), 
    s_all.std().item())
    
    
# norm_params = normalize_intrinsics(K_list=K_list,
#                                     widths=widths, heights =heights )
# print(norm_params)

#Plot graph of each parameter
# plt.figure(figsize=(10, 6))
# print(norm_params.shape)
# for i in range(norm_params.shape[1]):
#     plt.plot(norm_params[:, i], label=f"Param {i+1}")
# plt.xlabel("Frame Index")
# plt.ylabel("Normalized Value")
# plt.title("Normalized Intrinsics Parameters")
# plt.legend(["fx/W", "fy/H", "cx/W", "cy/H", "skew/W"])
# plt.grid()
# output_path = "normalized_intrinsics_parameters.png"
# plt.savefig(output_path)
# print(f"Graph saved to {output_path}")

# names = ["fx/W","fy/H","cx/W","cy/H","skew/W"]
# for i, name in enumerate(names[:norm_params.shape[1]]):
#     stats = robust_stats(norm_params[:, i])
#     print(name, stats["median"], stats["cv"])
#     # flag outliers:
#     bad = np.where(np.abs(stats["z"]) > 3)[0]
#     if len(bad):
#         print(f"Outliers in {name}: frames {bad.tolist()}")


# data = np.load(save_path / "000001.npz")

# print(data['K'])

    
