#!/usr/bin/env python3
"""
Projective point-to-plane ICP VO for KITTI-style sequences using MoGe‑2 outputs.

Assumptions
- You already dumped per-frame MoGe‑2 outputs as NPZ files, one per frame, named like
    seq_dir/frame_000000.npz, frame_000001.npz, ...
- Each NPZ contains (at least):
    points : float32 [H,W,3]   3D points in camera coords (meters)
    normals: float32 [H,W,3]   surface normals (optional but recommended)
    mask   : uint8  [H,W]      1 = valid, 0 = invalid
    K      : float32 [3,3]     camera intrinsics for that frame (pixels)
  (depth/intrinsics per-pixel not needed here)

What it does
- For each pair (t -> t+1), estimates relative pose with projective point-to-plane ICP.
- Multi-scale pyramid (coarse->fine), robust (Huber) + simple correspondence filtering.
- Accumulates global trajectory (poses in cam_0 frame) and writes KITTI-style poses.txt.

Usage
  python icp_vo.py \
      --seq_dir /path/to/your/npz_sequence \
      --out_poses poses.txt \
      --levels 3 --iters 10 7 5 \
      --subsample 4 --max_corr 60000

Notes
- Keep intrinsics consistent across a sequence if your \"K\" varies slightly. You may
  pass --force_first_K to use frame 0 intrinsics for all frames.
- If normals are missing, the code will fall back to point-to-point (less stable).
- This is a reference implementation with clear math; optimize further for speed.
"""

from __future__ import annotations
import argparse
import glob
import os
from dataclasses import dataclass
import numpy as np
import open3d as o3d


# add with your imports
import cv2

def np_to_o3d_pcd(points_hw3: np.ndarray,
                  mask_hw: np.ndarray,
                  normals_hw3: np.ndarray | None = None,
                  voxel_size: float | None = None) -> o3d.geometry.PointCloud:
    """
    Convert dense [H,W,3] arrays (and mask) into an Open3D point cloud.
    Optionally voxel-downsample.
    """
    H, W, _ = points_hw3.shape
    m = mask_hw.astype(bool).reshape(-1)
    P = points_hw3.reshape(-1, 3)[m]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if normals_hw3 is not None:
        N = normals_hw3.reshape(-1, 3)[m]
        # normalize (safer for ICP point-to-plane)
        n_norm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        pcd.normals = o3d.utility.Vector3dVector(N / n_norm)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def icp_o3d_multiscale(src_frame: Frame,
                       tgt_frame: Frame,
                       T_init: np.ndarray | None = None,
                       voxel_sizes=(0.8, 0.4, 0.2),     # meters; coarse -> fine (KITTI-ish)
                       iters=(50, 30, 14),
                       max_corr_factor=1.5,
                       robust_delta=0.05,               # meters (Huber)
                       use_point_to_plane=True) -> np.ndarray:
    """
    Multi-scale 3D ICP using Open3D. Returns T such that T @ src ~= tgt.
    """
    assert len(voxel_sizes) == len(iters)
    T = np.eye(4) if T_init is None else T_init.copy()

    # Choose estimation method
    if use_point_to_plane and src_frame.normals is not None and tgt_frame.normals is not None:
        # kernel = o3d.pipelines.registration.RobustKernel(
            # o3d.pipelines.registration.TukeyLoss(k=robust_delta)
        # )
        loss = o3d.pipelines.registration.TukeyLoss(k=robust_delta)
        # estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)

        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)  
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    for vx, n_iter in zip(voxel_sizes, iters):
        # Build/downsample pcds at this scale
        pcd_s = np_to_o3d_pcd(src_frame.points, src_frame.mask, src_frame.normals, voxel_size=vx)
        pcd_t = np_to_o3d_pcd(tgt_frame.points, tgt_frame.mask, tgt_frame.normals, voxel_size=vx)

        # Transform source by current estimate
        pcd_s = pcd_s.transform(T)

        # Max correspondence distance ~ 1.5 * voxel size (tune if needed)
        max_corr = max(vx * max_corr_factor, 0.05)

        result = o3d.pipelines.registration.registration_icp(
            pcd_s, pcd_t, max_corr, np.eye(4),
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(n_iter))
        )

        # Update T (compose delta on the left because we already transformed pcd_s by T)
        T = result.transformation @ T

    return T


# ---------- RANSAC helpers ----------
def orb_matches(gray_prev, gray_curr, max_kp=4000, ratio=0.8):
    orb = cv2.ORB_create(nfeatures=max_kp, fastThreshold=7, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)
    if des1 is None or des2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for (m, n) in raw if m.distance < ratio * n.distance]
    if len(good) < 20:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])  # (N,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])  # (N,2)
    return pts1, pts2

def sample_3d_from_prev(points_prev_bhwc, mask_prev_hw, uv_prev, zmin=0.5, zmax=80.0):
    H, W, _ = points_prev_bhwc.shape
    u = np.clip(np.round(uv_prev[:,0]).astype(int), 0, W-1)
    v = np.clip(np.round(uv_prev[:,1]).astype(int), 0, H-1)
    P = points_prev_bhwc[v, u]                    # (N,3)
    valid = (mask_prev_hw[v, u] > 0)
    valid &= np.isfinite(P).all(axis=1)
    valid &= (P[:,2] > zmin) & (P[:,2] < zmax)
    return P[valid], valid

def pnp_ransac(P3D, uv_curr, K_pixels, reproj_px=2.5, iters=2000, conf=0.999):
    if len(P3D) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        P3D.astype(np.float64), uv_curr.astype(np.float64), K_pixels.astype(np.float64), None,
        iterationsCount=iters, reprojectionError=reproj_px, confidence=conf,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inliers is None or len(inliers) < 20:
        return None
    inl = inliers.reshape(-1)
    ok2, rvec, tvec = cv2.solvePnP(
        P3D[inl].astype(np.float64), uv_curr[inl].astype(np.float64), K_pixels.astype(np.float64), None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok2:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t, inl


# --------------------------- Small SE3 utilities --------------------------- #

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=v.dtype)

def rot_angle_deg(R: np.ndarray) -> float:
    """Return rotation angle (degrees) of a 3x3 rotation matrix."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from twist xi=[w(3), v(3)] to 4x4 SE3.
    Uses Rodrigues for rotation; stable for small |w|.
    """
    w = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(w)
    I = np.eye(3)
    if theta < 1e-9:
        R = I + skew(w)
        V = I + 0.5 * skew(w)
    else:
        k = w / theta
        K = skew(k)
        st = np.sin(theta)
        ct = np.cos(theta)
        R = I * ct + (1 - ct) * np.outer(k, k) + st * K
        # Right Jacobian for SO(3)
        A = st / theta
        B = (1 - ct) / (theta ** 2)
        C = (theta - st) / (theta ** 3)
        V = I * A + B * skew(w) + C * (skew(w) @ skew(w))
    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# --------------------------- Image pyramid helpers ------------------------ #

@dataclass
class Frame:
    points: np.ndarray  # [H,W,3]
    normals: np.ndarray | None  # [H,W,3] or None
    gray: np.ndarray | None  # [H,W] or None
    mask: np.ndarray    # [H,W] uint8
    K: np.ndarray       # [3,3]


def downsample_points(P: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2x downsample by averaging valid points; mask is AND/any."""
    H, W, _ = P.shape
    H2, W2 = H // 2, W // 2
    P2 = np.zeros((H2, W2, 3), dtype=P.dtype)
    M2 = np.zeros((H2, W2), dtype=M.dtype)
    for y2 in range(H2):
        y = 2 * y2
        for x2 in range(W2):
            x = 2 * x2
            m = M[y:y+2, x:x+2]
            if m.any():
                pts = P[y:y+2, x:x+2][m.astype(bool)]
                P2[y2, x2] = pts.mean(axis=0)
                M2[y2, x2] = 1
    return P2, M2


def downsample_normals(N: np.ndarray | None, M: np.ndarray) -> np.ndarray | None:
    if N is None:
        return None
    H, W, _ = N.shape
    H2, W2 = H // 2, W // 2
    N2 = np.zeros((H2, W2, 3), dtype=N.dtype)
    for y2 in range(H2):
        y = 2 * y2
        for x2 in range(W2):
            x = 2 * x2
            m = M[y:y+2, x:x+2]
            if m.any():
                ns = N[y:y+2, x:x+2][m.astype(bool)]
                n = ns.mean(axis=0)
                n_norm = np.linalg.norm(n) + 1e-9
                N2[y2, x2] = n / n_norm
    return N2


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def make_pyramid(frame: Frame, levels: int) -> list[Frame]:
    pyr: list[Frame] = []
    P, N, M, K, G = frame.points, frame.normals, frame.mask, frame.K, frame.gray
    H, W = P.shape[:2]
    sx = sy = 1.0
    for _ in range(levels):
        pyr.append(Frame(P, N, G, M, scale_K(K, sx, sy)))
        if P.shape[0] < 40 or P.shape[1] < 40:
            break
        M_prev = M.copy()
        P, M = downsample_points(P, M_prev)
        N = downsample_normals(N, M_prev)
        sx *= 0.5
        sy *= 0.5
    return pyr

# --------------------------- Correspondence & ICP ------------------------- #

@dataclass
class ICPParams:
    max_corr: int = 60000
    subsample: int = 4      # grid step at finest level; multiplied at coarser levels
    iters_per_level: tuple[int, ...] = (10, 7, 5)
    huber_delta: float = 0.05  # meters
    depth_clip: float = 80.0
    reject_z_min: float = 0.05
    reject_dist_max: float = 0.2  # meters (adaptive per level)
    damping: float = 1e-6


def project(K: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points P[N,3] to pixels; returns (uv[N,2], valid_mask[N])."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z = P[:, 2]
    valid = z > 1e-6
    u = fx * (P[:, 0] / (z + 1e-9)) + cx
    v = fy * (P[:, 1] / (z + 1e-9)) + cy
    uv = np.stack([u, v], axis=-1)
    return uv, valid


def bilinear_sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample 2D/3D image at floating uv (N,2). Returns (N, C or 1)."""
    H, W = img.shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    w00 = (x1 - x) * (y1 - y)
    w01 = (x1 - x) * (y - y0)
    w10 = (x - x0) * (y1 - y)
    w11 = (x - x0) * (y - y0)

    x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)

    if img.ndim == 2:
        c00 = img[y0, x0]
        c01 = img[y1, x0]
        c10 = img[y0, x1]
        c11 = img[y1, x1]
        return (w00*c00 + w01*c01 + w10*c10 + w11*c11).reshape(-1, 1)
    else:
        c00 = img[y0, x0, :]
        c01 = img[y1, x0, :]
        c10 = img[y0, x1, :]
        c11 = img[y1, x1, :]
        return (w00[:,None]*c00 + w01[:,None]*c01 + w10[:,None]*c10 + w11[:,None]*c11)


def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    mask = a > delta
    w[mask] = (delta / (a[mask] + 1e-12))
    return w


def icp_point_to_plane(src: Frame, tgt: Frame, params: ICPParams) -> np.ndarray:
    """Estimate T (4x4) that takes src -> tgt using projective ICP."""
    T = np.eye(4, dtype=np.float64)

    # Determine grid subsampling based on level (we call this on each level already)
    H, W = src.points.shape[:2]
    ys = np.arange(0, H, params.subsample)
    xs = np.arange(0, W, params.subsample)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

    # Precompute 3D source points & validity
    Msrc = src.mask.astype(bool)
    valid_grid = Msrc[grid[:,1], grid[:,0]]
    grid = grid[valid_grid]
    Ps = src.points[grid[:,1], grid[:,0]].astype(np.float64)
    if Ps.shape[0] == 0:
        return T

    # Optionally cap number of correspondences
    if Ps.shape[0] > params.max_corr:
        idx = np.random.choice(Ps.shape[0], params.max_corr, replace=False)
        Ps = Ps[idx]
        grid = grid[idx]

    use_normals = tgt.normals is not None

    for it in range(max(1, params.iters_per_level[-1])):  # actual iters controlled outside
        # Transform source points with current T
        R = T[:3, :3]
        t = T[:3, 3]
        Ps_w = (R @ Ps.T).T + t
        # Reject bad Z
        zmask = (Ps_w[:, 2] > params.reject_z_min) & (Ps_w[:, 2] < params.depth_clip)
        Ps_w = Ps_w[zmask]
        if Ps_w.shape[0] < 1000:
            break

        # Project to target pixels
        uv, vproj = project(tgt.K, Ps_w)
        # Keep pixels inside image
        Ht, Wt = tgt.points.shape[:2]
        inside = (uv[:,0] >= 0) & (uv[:,0] <= Wt-1) & (uv[:,1] >= 0) & (uv[:,1] <= Ht-1)
        keep = vproj & inside
        if keep.sum() < 1000:
            break
        Ps_w = Ps_w[keep]
        uv = uv[keep]

        # Fetch target mask; reject invalid
        Mtg = bilinear_sample(tgt.mask.astype(np.float64), uv)[:,0] > 0.5
        if Mtg.sum() < 1000:
            break
        Ps_w = Ps_w[Mtg]
        uv = uv[Mtg]

        # Fetch target points & normals
        Pt = bilinear_sample(tgt.points.astype(np.float64), uv)  # [N,3]
        if use_normals:
            Nt = bilinear_sample(tgt.normals.astype(np.float64), uv)  # [N,3]
            # Normalize normals (robustness to interp)
            n_norm = np.linalg.norm(Nt, axis=1, keepdims=True) + 1e-9
            Nt = Nt / n_norm
        else:
            Nt = None

        # Residuals
        if use_normals:
            r = np.sum(Nt * (Ps_w - Pt), axis=1)  # [N]
        else:
            r = np.linalg.norm(Ps_w - Pt, axis=1)

        # Reject far correspondences (adaptive)
        dist_thresh = params.reject_dist_max
        mclose = np.abs(r) < dist_thresh
        Ps_w = Ps_w[mclose]
        Pt = Pt[mclose]
        if use_normals:
            Nt = Nt[mclose]
            r = r[mclose]
        else:
            r = np.linalg.norm(Ps_w - Pt, axis=1)

        if Ps_w.shape[0] < 500:
            break

        # Jacobians for point-to-plane: dr/dxi = n^T [ -[p']_x | I ]
        if use_normals:
            Np = Nt  # [N,3]
            Px = Ps_w  # [N,3]
            # Build J efficiently
            # For each i: Ji (1x6) = [ n^T * -[p']_x , n^T ]
            # -[p']_x * w = -(p' x w) => n^T (-(p' x w)) = (p' x n)^T w
            # cross = np.cross(Px, Np)  # [N,3] equals -(p' x n)? Careful:
            # n^T * (-[p']_x * w) = - n^T (p' x w) = (n x p')^T w
            # We'll use (n x p') for rotational part
            # print(f"cross shape: {Px.shape}, Np shape: {Np.shape}")  # Debug print

            rot = np.cross(Np, Px)  # [N,3]
            J = np.concatenate([rot, Np], axis=1)  # [N,6]
        else:
            # Point-to-point linearization (less stable): r = ||p'-pt||; directional Jacobian
            # Use unit direction u = (p'-pt)/||.||
            d = Ps_w - Pt
            norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
            u = d / norm
            rot = np.cross(u, Ps_w)
            J = np.concatenate([rot, u], axis=1)

        w = huber_weights(r, params.huber_delta)
        W = w  # diagonal weights

        A = (J.T * W) @ J + params.damping * np.eye(6)
        b = (J.T * W) @ (-r)
        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dx) < 1e-9:
            break
        T = se3_exp(dx) @ T

    return T

# --------------------------- Sequence runner ------------------------------ #

def load_frame(npz_path: str, K_kitti, args) -> Frame:
    data = np.load(npz_path)
    points = data["points"]
    mask = data["mask"]
    normals = data["normals"] if "normals" in data else None
    if K_kitti is None:
        K = data["K"]
        K = scale_K(K, points.shape[1], points.shape[0])
    else:
        K = K_kitti
    
    # K = scale_K(K, points.shape[1], points.shape[0])
    
    return Frame(points=points, normals=normals,gray = data['gray'], mask=mask, K=K)


def write_kitti_poses(poses: list[np.ndarray], out_path: str):
    with open(out_path, "w") as f:
        for T in poses:
            line = " ".join([f"{v:.6f}" for v in T[:3, :].reshape(-1)])
            f.write(line + "\n")

def load_kitti_K(calib_txt: str) -> np.ndarray:
    """
    Read KITTI Odometry calib.txt and return cam0 intrinsics (pixels).
    Uses the P0 3x4 projection matrix: K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
    """
    with open(calib_txt, "r") as f:
        for line in f:
            if line.startswith("P0:"):
                vals = list(map(float, line.strip().split()[1:]))
                if len(vals) != 12:
                    raise RuntimeError("P0 line in calib.txt does not have 12 values")
                P0 = np.array(vals, dtype=np.float64).reshape(3, 4)
                fx, fy, cx, cy = P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]
                K = np.array([[fx, 0.0, cx],
                              [0.0, fy, cy],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
                return K
    raise RuntimeError("Could not find P0 in calib.txt")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True, help="Directory with frame_*.npz files")
    ap.add_argument("--out_poses", required=True, help="Output KITTI-style poses.txt")
    ap.add_argument("--levels", type=int, default=3, help="Number of pyramid levels")
    ap.add_argument("--iters", type=int, nargs="*", default=[10,7,5], help="Iters per level (coarse->fine)")
    ap.add_argument("--subsample", type=int, default=4, help="Grid step at finest level")
    ap.add_argument("--max_corr", type=int, default=60000, help="Max correspondences per level")
    ap.add_argument("--use_first_K", action="store_true",default=False, help="Force intrinsics from first frame for all")
    args = ap.parse_args()

    # first = True

    files = sorted(glob.glob(os.path.join(args.seq_dir, "*.npz")))
    if len(files) < 2:
        raise RuntimeError("Need at least two frames.")
    
    # K_fixed = load_kitti_K(os.path.expanduser("~/datasets/kitti/dataset/sequences/00/calib.txt"))
    K_fixed = None
    # Load first frame
    f0 = load_frame(files[0], K_fixed, args)

    if args.use_first_K:
        K_fixed = f0.K


    # params = ICPParams(max_corr=args.max_corr, subsample=args.subsample,
                    #    iters_per_level=tuple(args.iters))

    poses = [np.eye(4)]
    step_lengths = []
    rot_angles = []


    for i in range(len(files)-1):
        # if i>=100:
            # break
        
        src = load_frame(files[i],K_fixed, args)
        tgt = load_frame(files[i+1], K_fixed, args)

        gray_prev = src.gray
        gray_curr = tgt.gray
       
        if gray_prev is not None and gray_curr is not None:
            # Resize grays to the SAME resolution as your point maps to avoid rescaling keypoints.
            Hp, Wp = src.points.shape[:2]
            if gray_prev.shape[:2] != (Hp, Wp):
                print(f"Resizing gray_prev {gray_prev.shape[:2]} -> {(Hp, Wp)}")
                gray_prev = cv2.resize(gray_prev, (Wp, Hp), interpolation=cv2.INTER_AREA)
            if gray_curr.shape[:2] != (Hp, Wp):
                print(f"Resizing gray_curr {gray_curr.shape[:2]} -> {(Hp, Wp)}")
                gray_curr = cv2.resize(gray_curr, (Wp, Hp), interpolation=cv2.INTER_AREA)

            m = orb_matches(gray_prev, gray_curr)
            if m is not None:
                uv_prev, uv_curr = m
                # Sample 3D from prev points at matched pixels
                P3D, vmask = sample_3d_from_prev(src.points, src.mask, uv_prev, zmin=0.5, zmax=80.0)
                uv_curr = uv_curr[vmask]
                rt = pnp_ransac(P3D, uv_curr, tgt.K, reproj_px=2.5, iters=2000, conf=0.999)
                if rt is not None:
                    R_pnp, t_pnp, _ = rt
                    T = np.eye(4, dtype=np.float64)
                    T[:3,:3] = R_pnp
                    T[:3, 3] = t_pnp
                else:
                    T = np.eye(4, dtype=np.float64)
            else:
                T = np.eye(4, dtype=np.float64)
        else:
            # If we can't find images, skip RANSAC (ICM behaves as before).
            T = np.eye(4, dtype=np.float64)





        # Build pyramids (coarse->fine). Ensure iters list matches number of levels we built.
        # pyr_src = make_pyramid(src, args.levels)
        # pyr_tgt = make_pyramid(tgt, args.levels)
        # n_levels = min(len(pyr_src), len(pyr_tgt))
        # iters = list(params.iters_per_level)
        # if len(iters) < n_levels:
        #     iters = [iters[0]]*(n_levels-1) + [iters[-1]]
        # elif len(iters) > n_levels:
        #     iters = iters[:n_levels]

        # # T = np.eye(4)
        # for lvl in range(n_levels):
        #     # Coarsest first
        #     L = n_levels - 1 - lvl
        #     s = ICPParams(
        #         max_corr=params.max_corr,
        #         subsample=params.subsample * (2 ** L),
        #         iters_per_level=(iters[L],),
        #         huber_delta=params.huber_delta * (1.5 ** L),
        #         depth_clip=params.depth_clip,
        #         reject_z_min=params.reject_z_min,
        #         reject_dist_max=params.reject_dist_max * (1.5 ** L),
        #         damping=params.damping,
        #     )
        #     # Transform src by current T and re-center? We incorporate T inside icp function by initializing T.
        #     # For simplicity, we left-multiply the incremental delta per level.
        #     T_lvl = icp_point_to_plane(
        #         Frame((T[:3,:3] @ pyr_src[L].points.reshape(-1,3).T).T.reshape(pyr_src[L].points.shape) + T[:3,3],
        #               pyr_src[L].normals,
        #               pyr_src[L].gray,
        #               pyr_src[L].mask,
        #               pyr_src[L].K),
        #         pyr_tgt[L],
        #         s,
        #     )
        #     T = T_lvl @ T


        T = icp_o3d_multiscale(
                src, tgt, T_init=T,
                voxel_sizes=(0.8, 0.4, 0.2),   # tune per dataset
                iters=(50, 30, 14),
                max_corr_factor=1.5,
                robust_delta=0.05,
                use_point_to_plane=True
            )

        # Per-frame debug stats
        step = float(np.linalg.norm(T[:3, 3]))
        angle = rot_angle_deg(T[:3, :3])
        step_lengths.append(step)
        rot_angles.append(angle)
        # if args.debug_steps:
        med = float(np.median(step_lengths))
        avg = float(np.mean(step_lengths))
        print(f"Frame {i}->{i+1}: |t|={step:.3f} m, rot={angle:.2f} deg | "
                f"running median={med:.3f} m, mean={avg:.3f} m")

        poses.append(poses[-1] @ T)
        # print(f"Frame {i}->{i+1}: correspondences OK, pose increment:\n{T}")

    write_kitti_poses(poses[1:], args.out_poses)  # KITTI expects one line per frame starting at 1
    print(f"Saved poses to {args.out_poses}")


if __name__ == "__main__":
    main()
