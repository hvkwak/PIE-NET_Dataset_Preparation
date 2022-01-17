import open3d
import numpy as np
import scipy.io as sio
from scipy.special import softmax
from scipy.interpolate import splprep, splev


def main():

    ref_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/5.mat')
    my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/3.mat')

    for i in range(64):
        down_sample_point = ref_mat['Training_data'][i, 0]['down_sample_point'][0, 0]
        #closed_curves = ref_mat['Training_data'][0, 0]['closed_gt_256_64_idx'][0, 0] 
        #view_point_4(down_sample_point)
        #view_point_4(down_sample_point[closed_curves][203, : ,:])
        #view_point_4(np.concatenate(down_sample_point[closed_curves]))
        view_point_4(down_sample_point)
        print()
    #test_pred_99_mat = sio.loadmat('/raid/home/hyovin.kwak/PIE-NET/main/test_result/test_pred_99.mat')
    #test_pred_99_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/test_pred_99.mat')
    #i = 0
    '''
    i = 0
    down_sample_point = test_pred_99_mat['input_point_cloud'][i, 0]['down_sample_point'][0, 0]
    edge_points_label_i = test_pred_99_mat['labels_edge_p'][i, :]
    corner_points_label_i = test_pred_99_mat['labels_corner_p'][i, :]
    pred_edge_points_label_i = softmax(test_pred_99_mat['pred_labels_edge_p'][i, :], axis = 1)
    pred_corner_points_label_i = softmax(test_pred_99_mat['pred_labels_corner_p'][i, :], axis = 1)
    pred_edge_points_label_i = pred_edge_points_label_i[:, 1] > 0.8
    pred_corner_points_label_i = pred_corner_points_label_i[:, 1] > 0.9

    # edge points prediction, 
    # green: true positives
    # blue: false negatives
    # red: false positives
    # edge
    # view_point_1(down_sample_point, edge_points_label_i, pred_edge_points_label_i)
    # corner
    # view_point_1(down_sample_point, corner_points_label_i, pred_corner_points_label_i)

    # with regression corrections
    # blue: without correction
    # red: with correction
    # view_point_2(down_sample_point, test_pred_99_mat['pred_reg_edge_p'][i, :], edge_points_label_i)
    # view_point_2(down_sample_point, test_pred_99_mat['pred_reg_corner_p'][i, :], corner_points_label_i)
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
    
    '''
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

if __name__ == "__main__": 
    main()