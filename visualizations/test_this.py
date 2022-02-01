import open3d
import numpy as np
from functools import partial


if __name__ == "__main__":
    
    def update_visualization(vis, arrayB):
        #print(point_cloud)
        #print(vis)
        #point_cloud.colors = open3d.utility.Vector3dVector(np.random.uniform(low = 0.0, high = 1.0, size = 150).reshape(50, 3))
        print(arrayB)
        arrayA = point_cloud.points
        arrayA = np.concatenate([arrayA, np.random.uniform(low = 0.0, high = 1.0, size = 3).reshape(1, 3)])
        point_cloud.points = open3d.utility.Vector3dVector(arrayA)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.run()

    arrayA = np.random.uniform(low = 0.0, high = 1.0, size = 150).reshape(50, 3)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(arrayA)
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 640, height = 640)
    vis.register_key_callback(257, partial(update_visualization, arrayB = arrayA))
    vis.add_geometry(point_cloud)
    vis.run()
    