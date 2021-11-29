import open3d
import numpy as np
import scipy.io as sio

def main():    
    ref_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/5.mat')['Training_data']
    my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/0042_4.mat')['Training_data']

    ref_down_sample_point = ref_mat[0, 0]['down_sample_point'][0, 0]
    ref_PC_8096_edge_points_label_bin = np.where(ref_mat[0, 0]['PC_8096_edge_points_label_bin'][0, 0][:, 0] == 1)[0]
    view_point(ref_down_sample_point)
    view_point(ref_down_sample_point[ref_PC_8096_edge_points_label_bin, ])

    my_down_sample_point = my_mat[0, 0]['down_sample_point'][0, 0]
    ref_PC_8096_edge_points_label_bin = np.where(my_mat[0, 0]['edge_points_label'][0, 0][0,:] == 1)[0]
    view_point(my_down_sample_point)
    view_point(my_down_sample_point[ref_PC_8096_edge_points_label_bin, ])
    

def view_point(points):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    point_cloud.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__": 
    main()