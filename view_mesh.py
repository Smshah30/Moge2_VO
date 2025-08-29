# view_mesh.py
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("frame_0005_mesh.ply")
mesh.compute_vertex_normals()
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

o3d.visualization.draw_geometries([mesh, axes], window_name="MoGe2 mesh", width=1280, height=800)
