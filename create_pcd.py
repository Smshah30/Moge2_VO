#!/usr/bin/env python3
# pip install open3d numpy
import argparse, os, numpy as np, open3d as o3d

def load_npz(npz_path):
    d = np.load(npz_path)
    # accept common keys you might have saved
    P = d['points']
    if P is None:
        raise RuntimeError("No 'points' array found in npz (looked for points_metric_fullres/points_metric/points).")
    P = np.asarray(P)                    # [H,W,3], meters
    M = np.asarray(d.get("mask", np.ones(P.shape[:2], np.uint8))).astype(bool)
    N = d.get("normals")                 # [H,W,3] or None
    return P, M, (np.asarray(N) if N is not None else None)

def to_pointcloud(P_hw3, M_hw, N_hw3=None, voxel=None):
    H, W = P_hw3.shape[:2]
    m = M_hw.reshape(-1)
    P = P_hw3.reshape(-1, 3)[m]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if N_hw3 is not None:
        N = N_hw3.reshape(-1, 3)[m]
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
        pcd.normals = o3d.utility.Vector3dVector(N)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    # ensure normals exist (Poisson/pt2plane benefits)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3*(voxel or 0.05), max_nn=30)
        )
    pcd.orient_normals_towards_camera_location([0,0,0])  # points are in camera coords
    return pcd

def poisson_mesh(pcd, depth=10, trim=0.6):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    thr = np.quantile(dens, trim)  # drop low-density verts
    keep = dens > thr
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices(); mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles(); mesh.remove_non_manifold_edges()
    return mesh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_npz", required=True, help="Path to frame .npz containing metric points")
    ap.add_argument("--out_pcd", default=None, help="Output .ply for point cloud (default: <stem>_pc.ply)")
    ap.add_argument("--mesh", action="store_true", help="Also save a Poisson mesh")
    ap.add_argument("--out_mesh", default=None, help="Output .ply for mesh (default: <stem>_mesh.ply)")
    ap.add_argument("--voxel", type=float, default=0.05, help="Voxel size in meters (e.g., 0.05)")
    ap.add_argument("--poisson_depth", type=int, default=10, help="Poisson octree depth (8–12 ok)")
    ap.add_argument("--poisson_trim", type=float, default=0.6, help="Trim fraction [0–1] for low-density verts")
    args = ap.parse_args()

    stem = os.path.splitext(args.frame_npz)[0]
    out_pcd = args.out_pcd or f"{stem}_pc.ply"
    out_mesh = args.out_mesh or f"{stem}_mesh.ply"

    P, M, N = load_npz(args.frame_npz)
    pcd = to_pointcloud(P, M, N, voxel=args.voxel)
    o3d.io.write_point_cloud(out_pcd, pcd, write_ascii=False)
    print(f"✅ point cloud → {out_pcd} ({np.asarray(pcd.points).shape[0]} pts)")

    if args.mesh:
        mesh = poisson_mesh(pcd, depth=args.poisson_depth, trim=args.poisson_trim)
        o3d.io.write_triangle_mesh(out_mesh, mesh)
        print(f"✅ mesh → {out_mesh} (V={len(mesh.vertices)}, F={len(mesh.triangles)})")

if __name__ == "__main__":
    main()
