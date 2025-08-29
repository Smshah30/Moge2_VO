#!/usr/bin/env python3
"""
Evaluate a monocular VO trajectory (from icp_vo.py) against KITTI Odometry ground truth.

Inputs
- GT poses: KITTI format poses file (e.g., data_odometry_poses/poses/00.txt),
  each line has 12 numbers (3x4) giving T_w_cam for frame i (0-based).
- EST poses: your output poses.txt from icp_vo.py, one 3x4 line per frame
  starting at frame 1 (i.e., T_0->t for t=1..N-1). We will pad Identity for frame 0.

What it does
- Aligns the estimated trajectory to GT with rigid SE(3) (no scale) or Sim(3) (with scale) using Umeyama.
- Computes ATE RMSE, ATE mean, and RPE (delta=1 frame) for translation and rotation.
- Plots XY trajectories and error histograms.

Usage
  python eval_kitti_vo.py \
    --gt_poses /path/to/KITTI/poses/00.txt \
    --est_poses /path/to/poses.txt \
    --align se3   # or sim3

Requires: numpy, matplotlib
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- I/O helpers ---------------------- #

def read_kitti_poses_txt(path: str) -> list[np.ndarray]:
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return poses


def read_est_poses_txt(path: str) -> list[np.ndarray]:
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return poses

# ---------------------- Alignment ----------------------- #

def umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool) -> tuple[np.ndarray, float, np.ndarray]:
    """Find R, s, t such that s*R*X + t ~ Y in least squares sense.
    X, Y: (3,N) point sets
    Returns (R, s, t)."""
    assert X.shape[0] == 3 and Y.shape[0] == 3
    muX = X.mean(axis=1, keepdims=True)
    muY = Y.mean(axis=1, keepdims=True)
    Xc = X - muX
    Yc = Y - muY
    Sigma = (Yc @ Xc.T) / X.shape[1]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        varX = (Xc ** 2).sum() / X.shape[1]
        s = np.trace(np.diag(D) @ S) / (varX + 1e-12)
    else:
        s = 1.0
    t = muY - s * R @ muX
    return R, s, t


def align_poses(gt_T: list[np.ndarray], est_T: list[np.ndarray], mode: str = 'se3') -> list[np.ndarray]:
    """Align est trajectory to gt by aligning positions only (Procrustes/Umeyama).
    mode: 'se3' (rigid, no scale) or 'sim3' (with scale).
    Returns aligned estimated poses (list of 4x4)."""
    N = min(len(gt_T), len(est_T))
    # Extract camera centers
    def centers(Ts):
        return np.stack([T[:3, 3] for T in Ts], axis=1)  # (3,N)
    X = centers(est_T[:N]).astype(np.float64)
    Y = centers(gt_T[:N]).astype(np.float64)
    R, s, t = umeyama(X, Y, with_scale=(mode == 'sim3'))
    aligned = []
    S = np.eye(4)
    S[:3, :3] = R * s
    S[:3, 3] = t.squeeze()
    for T in est_T[:N]:
        Ta = np.eye(4)
        Ta[:3, :3] = R @ T[:3, :3]
        Ta[:3, 3] = (s * (R @ T[:3, 3])) + t.squeeze()
        aligned.append(Ta)
    return aligned

# ---------------------- Metrics ------------------------ #

def pose_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def log_so3(R: np.ndarray) -> float:
    # return rotation angle in radians
    tr = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def ate_rmse(gt_T: list[np.ndarray], est_T: list[np.ndarray]) -> float:
    N = min(len(gt_T), len(est_T))
    gt = np.stack([gt_T[i][:3, 3] for i in range(N)], axis=0)
    est = np.stack([est_T[i][:3, 3] for i in range(N)], axis=0)
    err = np.linalg.norm(est - gt, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def rpe(gt_T: list[np.ndarray], est_T: list[np.ndarray], delta: int = 1) -> tuple[float, float]:
    """Relative Pose Error over a fixed delta.
    Returns (trans_rmse [m], rot_rmse [deg])."""
    trans_err = []
    rot_err = []
    N = min(len(gt_T), len(est_T))
    for i in range(N - delta):
        GT_rel = pose_inv(gt_T[i]) @ gt_T[i + delta]
        EST_rel = pose_inv(est_T[i]) @ est_T[i + delta]
        E = pose_inv(GT_rel) @ EST_rel
        t_err = np.linalg.norm(E[:3, 3])
        r_err = np.degrees(log_so3(E[:3, :3]))
        trans_err.append(t_err)
        rot_err.append(r_err)
    trans_rmse = float(np.sqrt(np.mean(np.array(trans_err) ** 2)))
    rot_rmse = float(np.sqrt(np.mean(np.array(rot_err) ** 2)))
    return trans_rmse, rot_rmse

# ---------------------- Plotting ----------------------- #

def plot_trajectories(gt_T: list[np.ndarray], est_T: list[np.ndarray], title: str, args):
    est = np.stack([T[:3, 3] for T in est_T], axis=0)
    plt.figure()
    if args.gt_poses:
        gt = np.stack([T[:3, 3] for T in gt_T], axis=0)
        plt.plot(gt[:, 0], gt[:, 2], label='GT')
    plt.plot(est[:, 0], est[:, 2], label='EST', linestyle='--')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title(title)
    plt.legend()
    if args.save_plot:
        plt.savefig(args.save_plot)
        print(f"Trajectory plot saved to {args.save_plot}")
    else:
        plt.show()
        # plt.savefig("kitti_vo_trajectory_3d2.png")

# ---------------------- Main -------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_poses', required=False, help='KITTI GT poses file (poses/XX.txt)')
    ap.add_argument('--est_poses', required=True, help='Estimated poses.txt from icp_vo.py')
    ap.add_argument('--align', choices=['se3', 'sim3', 'none'], default='none', help='Alignment mode')
    ap.add_argument('--delta', type=int, default=1, help='Delta for RPE')
    ap.add_argument('--save_plot', type=str, default=None, help='Path to save trajectory plot')
    args = ap.parse_args()

    if args.gt_poses:
        gt = read_kitti_poses_txt(args.gt_poses)  # length N_gt
    est = read_est_poses_txt(args.est_poses)  # length N_est_minus1 (starts at frame 1)

    # Pad identity for frame 0 to align indices: est_T[0] = I
    est = [np.eye(4)] + est

    gt_rel = None

    # Trim to common length
    if args.gt_poses:
        N = min(len(gt), len(est))
        gt = gt[:N]
        est = est[:N]

        # Optionally transform GT to be relative to its first frame (so both are T_0->t)
        T0_inv = pose_inv(gt[0])
        gt_rel = [T0_inv @ T for T in gt]

    # Align estimated to GT
    if args.align in ('se3', 'sim3') and args.gt_poses:
        est_aligned = align_poses(gt_rel, est, mode=args.align)
    else:
        est_aligned = est
    ate = float('nan')
    if args.gt_poses:
        # Compute metrics
        # print(f"Evaluating {len(est_aligned)} frames against GT")
        ate = ate_rmse(gt_rel, est_aligned)
        rpe_t, rpe_r = rpe(gt_rel, est_aligned, delta=args.delta)

        print(f"ATE RMSE: {ate:.3f} m")
        print(f"RPE (delta={args.delta}): {rpe_t:.3f} m, {rpe_r:.3f} deg")

    plot_trajectories(gt_rel, est_aligned, title=f"Trajectory (ATE={ate:.2f} m)", args=args)
    # plt.show()

if __name__ == '__main__':
    main()
