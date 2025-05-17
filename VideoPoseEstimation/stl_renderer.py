import numpy as np
import cv2
from stl import mesh
import trimesh
def render_projected_stl_with_normal_zbuffer(img, stl_path, rvec, tvec, camera_matrix):
    your_mesh = mesh.Mesh.from_file(stl_path)
    rendered = img.copy()
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    triangles = []
    for i, tri in enumerate(your_mesh.vectors):
        center = tri.mean(axis=0)
        cam_space = R @ center + t
        depth = cam_space[2]
        triangles.append((depth, i, tri))
    triangles.sort(reverse=True)
    for _, i, triangle in triangles:
        imgpts, _ = cv2.projectPoints(triangle, rvec, tvec, camera_matrix, None)
        pts = np.int32(imgpts).reshape(-1, 2)
        normal = your_mesh.normals[i]
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        color = (
            int((normal[0] * 0.5 + 0.5) * 255),
            int((normal[1] * 0.5 + 0.5) * 255),
            int((normal[2] * 0.5 + 0.5) * 255)
        )
        cv2.fillConvexPoly(rendered, pts, color)
    blended = cv2.addWeighted(img, 0.3, rendered, 0.7, 0)
    return blended

def draw_projected_stl_edges(img, stl_path, rvec, tvec, camera_matrix, color=(0, 0, 0)):
    mesh_obj = trimesh.load(stl_path, process=True)
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = trimesh.util.concatenate([g for g in mesh_obj.geometry.values()])
    all_edges = mesh_obj.edges_sorted
    edges_sorted, edges_count = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = edges_sorted[edges_count == 1]
    vertices = mesh_obj.vertices
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.squeeze().astype(int)
    for edge in boundary_edges:
        pt1 = tuple(imgpts[edge[0]])
        pt2 = tuple(imgpts[edge[1]])
        cv2.line(img, pt1, pt2, (0, 150, 150), 20)
    return img
