import copy
import numpy as np
import open3d as o3d
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)