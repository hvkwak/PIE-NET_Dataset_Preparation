import open3d
import numpy as np
import scipy.io as sio
from scipy.interpolate import splprep, splev

def main():
    
    ref_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/5.mat')['Training_data']
    my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/0006_0.mat')['Training_data']
    
    ref_mat_down_sample_point_max = 0.0
    ref_mat_down_sample_point_min = 0.0
    my_mat_down_sample_point_max = 0.0
    my_mat_down_sample_point_min = 0.0

    for i in range(64):
        ref_down_sample_point = ref_mat[i, 0]['down_sample_point'][0, 0]
        ref_PC_8096_edge_points_label_bin = np.where(ref_mat[i, 0]['PC_8096_edge_points_label_bin'][0, 0][:, 0] == 1)[0]
        ref_corner_points_label_bin = np.where(ref_mat[i, 0]['corner_points_label'][0, 0][:, 0] == 1)[0]
        down_sample_point = my_mat[i, 0]['down_sample_point'][0, 0]
        edge_points_label = np.where(my_mat[i, 0]['edge_points_label'][0, 0][0,:] == 1)[0]
        corner_points_label = np.where(my_mat[i, 0]['corner_points_label'][0, 0][0,:] == 1)[0]

        print("max", i, "X: ", np.max(ref_down_sample_point[:, 0]))
        print("max", i, "Y: ", np.max(ref_down_sample_point[:, 1]))
        print("max", i, "Z: ", np.max(ref_down_sample_point[:, 2]))
        print("min", i, " ", np.min(ref_down_sample_point))
        ref_mat_down_sample_point_max += np.max(ref_down_sample_point)
        ref_mat_down_sample_point_min += np.min(ref_down_sample_point)
        my_mat_down_sample_point_max += np.max(down_sample_point)
        my_mat_down_sample_point_min += np.min(down_sample_point)

    ref_mat_down_sample_point_max = ref_mat_down_sample_point_max / 64.0
    ref_mat_down_sample_point_min = ref_mat_down_sample_point_min / 64.0
    my_mat_down_sample_point_max = my_mat_down_sample_point_max / 64.0
    my_mat_down_sample_point_min = my_mat_down_sample_point_min / 64.0

    print("****** BASIC STATISTICS ******" )
    print("ref down_sample_point in [", ref_mat_down_sample_point_min, ", ", ref_mat_down_sample_point_max, "]")
    print("my down_sample_point in [", my_mat_down_sample_point_min, ", ", my_mat_down_sample_point_max, "]")

    for i in range(0, 64):
        ref_down_sample_point = ref_mat[i, 0]['down_sample_point'][0, 0]
        ref_PC_8096_edge_points_label_bin = np.where(ref_mat[i, 0]['PC_8096_edge_points_label_bin'][0, 0][:, 0] == 1)[0]
        ref_corner_points_label_bin = np.where(ref_mat[i, 0]['corner_points_label'][0, 0][:, 0] == 1)[0]
        view_point_1(ref_down_sample_point, ref_PC_8096_edge_points_label_bin, ref_corner_points_label_bin)
    
        down_sample_point = my_mat[i, 0]['down_sample_point'][0, 0]
        edge_points_label = np.where(my_mat[i, 0]['edge_points_label'][0, 0][0,:] == 1)[0]
        corner_points_label = np.where(my_mat[i, 0]['corner_points_label'][0, 0][0,:] == 1)[0]
        view_point_1(down_sample_point, edge_points_label, corner_points_label)
        #print("corner_points_label: ", corner_points_label.shape)
    
def view_point_1(points, my_down_sample_point_edge, my_down_sample_point_corner):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)    
    color1 = [0.0, 0.99, 0.0] # BSpline one degree, green
    color2 = [0.0, 0.0, 0.99] # edge, blue
    color3 = [0.99, 0, 0.0] # corner, red
    #color4 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
    color_array = np.zeros_like(points)
    
    color_array[my_down_sample_point_edge, ] = color2
    color_array[my_down_sample_point_corner, ] = color3
    
    '''
    if len(BSpline_two_or_more_degree_list) > 2:
        color_array[BSpline_two_or_more_degree_list[0], ] = color1 # green
        color_array[BSpline_two_or_more_degree_list[1], ] = color2 # red
        color_array[BSpline_two_or_more_degree_list[2], ] = color3 # blue
    '''

    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    #point_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    open3d.visualization.draw_geometries([point_cloud])



def view_point(points, BSpline_per_degree_list):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)    
    color1 = [0.0, 0.99, 0.0] # BSpline one degree, green
    color2 = [0.0, 0.0, 0.99] # edge, blue
    color3 = [0.99, 0, 0.0] # corner, red
    color4 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
    color_array = np.zeros_like(points)
    '''
    color_array[my_down_sample_point_edge, ] = color2
    color_array[my_down_sample_point_corner, ] = color3
    '''
    k = 0
    while k < len(BSpline_per_degree_list):
        color_array[BSpline_per_degree_list[k][2], ] = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
        k = k + 1

    '''
    if len(BSpline_two_or_more_degree_list) > 2:
        color_array[BSpline_two_or_more_degree_list[0], ] = color1 # green
        color_array[BSpline_two_or_more_degree_list[1], ] = color2 # red
        color_array[BSpline_two_or_more_degree_list[2], ] = color3 # blue
    '''

    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    #point_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__": 
    main()