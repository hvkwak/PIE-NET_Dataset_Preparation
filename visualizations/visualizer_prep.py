import open3d
import numpy as np
import scipy.io as sio
from functools import partial

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
    mat_num = 3

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

        # arrayR and first visualization
        listB = [edge_points_label, corner_points_label]
        listG = [down_sample_point+edge_points_residual_vector, down_sample_point+corner_points_residual_vector]
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
        
        # create updates
        def close_visualization(vis):
            vis.close()

        k = 1
        def update_visualization31(vis, down_sample_point, listB, listG):
            global k
            # k just stands for k-th element in listB and listG
            assert len(listB) == len(listG)
            color1 = [0.5, 0.5, 0.5]   # gray
            color2 = [0.0, 0.0, 0.99] # blue
            color3 = [0.0, 0.99, 0.0] # green

            # arrayR and take first
            #arrayR = down_sample_point
            arrayB = listB[k]
            arrayG = listG[k]
            assert np.sum((down_sample_point[np.where(arrayB!=1)[0] ,] - arrayG[np.where(arrayB!=1)[0] ,])**2) == 0.0
            assert np.where(arrayB == 1)[0].max() < down_sample_point.shape[0]

            # colors
            arrayG = listG[k][np.where(arrayB == 1)[0], ]
            color_array = np.zeros_like(np.concatenate([down_sample_point, arrayG], axis = 0))
            color_array[:down_sample_point.shape[0], :] = color1
            color_array[np.where(arrayB == 1)[0], :] = color2
            color_array[down_sample_point.shape[0]:, ] = color3
            if k < len(listB)-1:
                k = k + 1
            point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, arrayG], axis = 0))
            point_cloud.colors = open3d.utility.Vector3dVector(color_array)
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
            #vis.run()
            
        k_open_curve = 0
        def update_visualization32(vis, \
                                down_sample_point, \
                                open_gt_pair_idx, \
                                open_gt_valid_mask, \
                                open_gt_256_64_idx, \
                                open_gt_type, \
                                open_gt_res, \
                                open_gt_sample_points, \
                                open_gt_mask):
            global k_open_curve

            # check
            open_gt_mask_sum = open_gt_mask[:, 0].sum()
            open_gt_valid_mask_sum = open_gt_valid_mask.sum()
            #print("num_open_curves: ", open_gt_valid_mask_sum)qqq
            #print("open_gt_mask_sum: ", open_gt_mask_sum)
            #print("current curve: ", k_open_curve)
            assert open_gt_mask_sum == open_gt_valid_mask_sum

            # color
            color1 = [0.5, 0.5, 0.5]   # gray
            color2 = [0.0, 0.0, 0.99] # blue
            color3 = [0.0, 0.99, 0.0] # green

            if k_open_curve < open_gt_mask_sum:
                #print(k_open_curve)
                #print("open_gt_256_64_idx[k_open_curve[0]]: ", open_gt_256_64_idx[k_open_curve])
                idx_equal = open_gt_pair_idx[k_open_curve] == np.array([open_gt_256_64_idx[k_open_curve][0], open_gt_256_64_idx[k_open_curve][-1]])
                points_close_all = np.allclose(down_sample_point[open_gt_256_64_idx[k_open_curve], ], open_gt_sample_points[k_open_curve, ])
                assert points_close_all
                assert idx_equal.all()
                
                color_array = np.zeros_like(down_sample_point)
                color_array[:, :] = color1
                color_array[open_gt_256_64_idx[k_open_curve, ], :] = color2
                color_array[open_gt_pair_idx[k_open_curve, ], :] = color3
                if k_open_curve < open_gt_mask_sum - 1:
                    k_open_curve = k_open_curve + 1
                point_cloud.points = open3d.utility.Vector3dVector(down_sample_point)
                point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()
                #vis.run()                

        # create point clouds and visualizers
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, arrayG], axis = 0))
        point_cloud.colors = open3d.utility.Vector3dVector(color_array)

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_callback(87, partial(update_visualization31, down_sample_point = down_sample_point, listB = listB, listG = listG)) # W    
        vis.register_key_callback(69, partial(update_visualization32, \
                                            down_sample_point = down_sample_point, \
                                            open_gt_pair_idx = open_gt_pair_idx, \
                                            open_gt_valid_mask = open_gt_valid_mask, \
                                            open_gt_256_64_idx = open_gt_256_64_idx, \
                                            open_gt_type = open_gt_type, \
                                            open_gt_res = open_gt_res, \
                                            open_gt_sample_points = open_gt_sample_points, \
                                            open_gt_mask = open_gt_mask)) # Q
        vis.register_key_callback(81, close_visualization) # Q
        vis.add_geometry(point_cloud)
        vis.run()

    vis.destroy_window()