# view_pcd.py
import open3d as o3d

pcd = o3d.io.read_point_cloud("frame_0005_pc.ply")  # your output
# optional: add a coordinate frame so you see axes
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

o3d.visualization.draw_geometries(
    [pcd, axes],
    window_name="MoGe2 point cloud",
    width=1280, height=800,
    point_show_normal=False
)
