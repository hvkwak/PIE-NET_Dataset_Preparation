import open3d
import numpy as np
import scipy.io as sio
from functools import partial

def main():

        # 3. Based on: open_gt_pair_idx & open_gt_valid_mask & open_gt_type & open_gt_mask
        #    Check:    open_gt_256_64_idx in down_sample_points
        #    ADD:      open_gt_res
        
        # Visualize down sampled points
        # make sure that it's a same type.
        '''
        visualize2(down_sample_point = down_sample_point, \
                   open_gt_pair_idx = open_gt_pair_idx, \
                   open_gt_valid_mask = open_gt_valid_mask, \
                   open_gt_type = open_gt_type, \
                   open_gt_mask = open_gt_mask, \
                   open_gt_256_64_idx = open_gt_256_64_idx, \
                   open_gt_res = open_gt_res)
        '''


        # Make sure that..
        # open_gt_sample_points == down_sample_points[open_gt_256_64_idx, ...]
        # open_gt_pair_idx == (open_gt_256_64_idx[0], open_gt_256_64_idx[-1])
        #
        # Use for loop and updates within Visualizer Class
        # http://www.open3d.org/docs/0.9.0/tutorial/Advanced/customized_visualization.html
        # http://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html
'''
def generate_color_array2():
    print()

def visualize2(**kwargs):

    # for gt mask:
        # Down Sampled points in red
        # open_gt_pair_idx blue
        # open_gt_256_64_idx in green
        # make sure unique(256_64_idx) == valid_mask
        # make sure idx beyond sum == 0
    arrayR = kwargs['down_sample_point']
    arrayB_idx = kwargs['open_gt_pair_idx'][0]
    arrayG_idx = kwargs['open_gt_256_64_idx'][0]
    kwargs['open_gt_256_64_idx']

    for i in range(1, kwargs['open_gt_mask'].shape[0]):
        if kwargs['open_gt_mask'][i][0] == 1:
'''




'''
def view_point_4(closed_curves):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(closed_curves)    
    #color1 = [0.0, 0.99, 0.0] # BSpline one degree, green
    #color2 = [0.0, 0.0, 0.99] # blue, GT.
    #color3 = [0.99, 0, 0.0] # red, prediction
    #color_array = np.zeros_like(closed_curves)
    #color_array[np.where(gt == 1)[0], ] = color2 # blue
    #color_array[np.where(pred == 1)[0], ] = color3 # red
    #color_array[np.intersect1d(np.where(gt == 1)[0], np.where(pred == 1)[0]), ] = color1 # green
    #point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    open3d.visualization.draw_geometries([point_cloud])


def view_point_2(down_sample_point, pred_reg, edge_points_label_i):
    down_sample_point2 = down_sample_point.copy()
    #down_sample_point[:, 0] = down_sample_point[:, 0] + 1
    points = np.concatenate([down_sample_point2, down_sample_point+pred_reg])
    partA = np.concatenate([edge_points_label_i, np.zeros_like(edge_points_label_i)])
    partB = np.concatenate([np.zeros_like(edge_points_label_i), edge_points_label_i])
    color2 = [0.0, 0.0, 0.99] # blue, without correction
    color3 = [0.99, 0, 0.0] # red, with correction
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    color_array = np.zeros_like(points)
    color_array[np.where(partA == 1)[0], ] = color2 # blue
    color_array[np.where(partB == 1)[0], ] = color3 # red
    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    open3d.visualization.draw_geometries([point_cloud])


def view_point_1(points, gt, pred):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)    
    color1 = [0.0, 0.99, 0.0] # BSpline one degree, green
    color2 = [0.0, 0.0, 0.99] # blue, GT.
    color3 = [0.99, 0, 0.0] # red, prediction
    color_array = np.zeros_like(points)
    color_array[np.where(gt == 1)[0], ] = color2 # blue
    color_array[np.where(pred == 1)[0], ] = color3 # red
    color_array[np.intersect1d(np.where(gt == 1)[0], np.where(pred == 1)[0]), ] = color1 # green
    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    open3d.visualization.draw_geometries([point_cloud])

def view_point(points, BSpline_per_degree_list):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)    
    #color1 = [0.0, 0.99, 0.0] # BSpline one degree, green
    #color2 = [0.0, 0.0, 0.99] # edge, blue
    #color3 = [0.99, 0, 0.0] # corner, red
    #color4 = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
    color_array = np.zeros_like(points)

    k = 0
    while k < len(BSpline_per_degree_list):
        color_array[BSpline_per_degree_list[k][2], ] = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
        k = k + 1
    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    #point_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    open3d.visualization.draw_geometries([point_cloud])
    
    
    i = 0
    my_mat = sio.loadmat('/raid/home/hyovin.kwak/PIE-NET_Dataset_Preparation/0.mat')['Training_data']
    down_sample_point = my_mat[i, 0]['down_sample_point'][0, 0]
    edge_points_label = np.where(my_mat[i, 0]['edge_points_label'][0, 0][0,:] == 1)[0]
    corner_points_label = np.where(my_mat[i, 0]['corner_points_label'][0, 0][0,:] == 1)[0]

    ref_mat = sio.loadmat('/raid/home/hyovin.kwak/PIE-NET/main/train_data/5.mat')['Training_data']
    ref_down_sample_point = ref_mat[i, 0]['down_sample_point'][0, 0]
    ref_edge_points_label = np.where(ref_mat[i, 0]['PC_8096_edge_points_label_bin'][0, 0][0,:] == 1)[0]
    ref_corner_points_label = np.where(ref_mat[i, 0]['corner_points_label'][0, 0][0,:] == 1)[0]
    #view_point_1(ref_down_sample_point, ref_edge_points_label, ref_corner_points_label)
    #print()
    
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
    '''

def update_visualization(vis, arrayR, listB, listG, k):

    assert len(listB) == len(listG)
    color1 = [0.5, 0.5, 0.5]   # gray
    color2 = [0.0, 0.0, 0.99] # blue
    color3 = [0.0, 0.99, 0.0] # green

    # arrayR and take first
    arrayR = down_sample_point
    arrayB = listB[k]
    arrayG = listG[k]
    assert np.sum((arrayR[np.where(arrayB!=1)[0] ,] - arrayG[np.where(arrayB!=1)[0] ,])**2) == 0.0
    assert np.where(arrayB == 1)[0].max() < arrayR.shape[0]

    # colors
    arrayG = listG[k][np.where(arrayB == 1)[0], ]
    color_array = np.zeros_like(np.concatenate([arrayR, arrayG], axis = 0))
    color_array[:arrayR.shape[0], :] = color1
    color_array[np.where(arrayB == 1)[0], :] = color2
    color_array[arrayR.shape[0]:, ] = color3    

    point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([arrayR, arrayG], axis = 0))
    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    k = k + 1

if __name__ == "__main__":
    
    # Task1
    # 1. R down_sample_point, 
    #    B edge_points_label, 
    #    G down_sample_point+edge_points_residual_vector = corrected edge points
    # 2. R down_sample_point, 
    #    B corner_points_label, 
    #    G down_sample_point+corner_points_residual_vector = corrected edge points
    my_mat = sio.loadmat('/home/hyobin/Documents/PIE-NET_Dataset_Preparation/visualizations/0.mat')
    color1 = [0.5, 0.5, 0.5]   # gray
    color2 = [0.0, 0.0, 0.99] # blue
    color3 = [0.0, 0.99, 0.0] # green
    mat_num = 10

    for i in range(0, mat_num):
        # sec 3.1.
        down_sample_point = my_mat['Training_data'][i, 0]['down_sample_point'][0, 0]
        edge_points_label = my_mat['Training_data'][i, 0]['edge_points_label'][0, 0]
        edge_points_residual_vector = my_mat['Training_data'][i, 0]['edge_points_residual_vector'][0, 0]
        corner_points_label = my_mat['Training_data'][i, 0]['corner_points_label'][0, 0]
        corner_points_residual_vector = my_mat['Training_data'][i, 0]['corner_points_residual_vector'][0, 0]
        
        # sec 3.2.
        open_gt_pair_idx = my_mat['Training_data'][i, 0]['open_gt_pair_idx'][0, 0]
        open_gt_valid_mask = my_mat['Training_data'][i, 0]['open_gt_valid_mask'][0, 0]
        open_gt_256_64_idx = my_mat['Training_data'][i, 0]['open_gt_256_64_idx'][0, 0]
        open_gt_type = my_mat['Training_data'][i, 0]['open_gt_type'][0, 0]
        open_gt_res = my_mat['Training_data'][i, 0]['open_gt_res'][0, 0]
        open_gt_sample_points = my_mat['Training_data'][i, 0]['open_gt_sample_points'][0, 0]
        open_gt_mask = my_mat['Training_data'][i, 0]['open_gt_mask'][0, 0]

        listB = [edge_points_label, corner_points_label]
        listG = [down_sample_point+edge_points_residual_vector, down_sample_point+corner_points_residual_vector]

        # arrayR and take first
        arrayR = down_sample_point
        arrayB = listB[0]
        arrayG = listG[0]
        assert np.sum((arrayR[np.where(arrayB!=1)[0] ,] - arrayG[np.where(arrayB!=1)[0] ,])**2) == 0.0
        assert np.where(arrayB == 1)[0].max() < arrayR.shape[0]

        # colors
        arrayG = listG[0][np.where(arrayB == 1)[0], ]
        color_array = np.zeros_like(np.concatenate([arrayR, arrayG], axis = 0))
        color_array[:arrayR.shape[0], :] = color1
        color_array[np.where(arrayB == 1)[0], :] = color2
        color_array[arrayR.shape[0]:, ] = color3
        
        # create visualizer
        k = 1
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, arrayG], axis = 0))
        point_cloud.colors = open3d.utility.Vector3dVector(color_array)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_callback(257, partial(update_visualization, arrayR = down_sample_point, listB = listB, listG = listG, k = k))
        vis.add_geometry(point_cloud)
        vis.run()