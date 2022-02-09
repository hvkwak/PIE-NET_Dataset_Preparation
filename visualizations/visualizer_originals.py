import numpy as np
import open3d
import os
import sys
from utils import *
from grafkom1Framework import ObjLoader
from functools import partial

# visualizes original curves

if __name__ == "__main__":

    # Arguments check
    args = sys.argv[1:]
    print("args: ", args)

    # open files, change FPS_num from (str) to (int), generate log file.
    list_obj_file = open(args[0], "r") # "/raid/home/hyovin.kwak/all/obj/${1}_list_obj.txt"
    list_ftr_file = open(args[1], "r") # "/raid/home/hyovin.kwak/all/obj/${1}_list_yml.txt"
    save_prefix = args[2] # $foldernum
    FPS_num = 8096
    log_dir = args[3]     # "/raid/home/hyovin.kwak/PIE-NET_Dataset_Preparation/log/"
    log_file_name = save_prefix+'generate_dataset_log.txt'
    log_fout = open(os.path.join(log_dir, log_file_name), 'w+')
    

    # readlines(of files) to make sure they are same length
    list_obj_lines = list_obj_file.readlines()
    list_ftr_lines = list_ftr_file.readlines()
    model_total_num = len(list_ftr_lines)
    assert model_total_num == len(list_obj_lines)
    
    batch_count = 0
    file_count = 0
    data = {'batch_count': batch_count, 'Training_data': np.zeros((64, 1), dtype = object)}
    take_sharp_false = True
    for i in range(model_total_num):
        
        model_name_obj = "_".join(list_obj_lines[i].split('/')[-1].split('_')[0:2])
        model_name_ftr = "_".join(list_ftr_lines[i].split('/')[-1].split('_')[0:2])
        model_name_obj = delete_newline(model_name_obj)
        model_name_ftr = delete_newline(model_name_ftr)
        list_obj_line = delete_newline(list_obj_lines[i])
        list_ftr_line = delete_newline(list_ftr_lines[i])
        skip_this_model = False
        
        if model_name_obj == model_name_ftr:
            # make sure that there's no "\n" in the line.
            print("Processing: ", model_name_ftr, \
                ".."+str(i+1) + "/" + str(model_total_num))
            log_string("Processing: "+ model_name_ftr+ ".."+str(i+1) + "/" + str(model_total_num), log_fout)

            # load the object file: all vertices / faces
            Loader = ObjLoader(list_obj_line)
            vertices = np.array(Loader.vertices)
            faces = np.array(Loader.faces)
            vertex_normals = np.array(Loader.vertex_normals)
            del Loader
            
            # Type1
            # make sure we have < 30K vertices to keep it simple.
            if vertices.shape[0] > 30000: 
                print("vertices:", vertices.shape[0], " > 40000. skip this.")
                #log_string("Type 1", log_fout)
                #log_string("vertices " +str(vertices.shape[0])+" > 40000. skip this.", log_fout)
                del vertices
                del faces
                del vertex_normals
                continue

            '''
            #Type2
            # Curves with vertex indices: (sharp and not sharp)edges of BSpline, Line, Cycle only.
            if not mostly_sharp_edges(list_ftr_line, threshold=0.30):
                print("sharp_true_count/(sharp_true_count+sharp_false_count) < 0.30. skip this.")
                log_string("Type 2", log_fout)
                log_string("sharp_true_count/(sharp_true_count+sharp_false_count) < 0.30. skip this.", log_fout)
                continue
            '''
            
            # Type3
            # This has curves other than Circle, BSpline or Line, skip this.
            all_curves = []
            try:
                all_curves = curves_with_vertex_indices(list_ftr_line, take_sharp_false=take_sharp_false)
            except:
                print("there are curves not in [Circle, BSpline, Line]. skip this.")
                #log_string("Type 3", log_fout)
                #log_string("there are curves not in [Circle, BSpline, Line]. skip this.", log_fout)
                continue                        
            curve_num = len(all_curves)

            # Preprocessing: (Re)classify / Filter points accordingly!
            BSpline_list = []
            Line_list = []
            Circle_list = []
            # Group all the curves.
            k = 0
            while k < curve_num:
                curve = all_curves[k]
                if curve[0] == 'BSpline': BSpline_list.append(curve)
                elif curve[0] == 'Line': Line_list.append(curve)
                elif curve[0] == 'Circle': Circle_list.append(curve)
                k = k + 1
    
    
        # Task1
        # 1. R down_sample_point, 
        #    B edge_points_label, 
        #    G down_sample_point+edge_points_residual_vector = corrected edge points
        # 2. R down_sample_point, 
        #    B corner_points_label, 
        #    G down_sample_point+corner_points_residual_vector = corrected edge points
        #my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/visualizations/0.mat')
        color1 = [0.5, 0.5, 0.5]   # gray
        color2 = [0.0, 0.0, 0.99] # blue
        color3 = [0.0, 0.99, 0.0] # green


        # create updates
        def close_visualization(vis):
            vis.close()

        k = 1 # 
        def update_visualization(vis, vertices, BSpline_list, Line_list, Circle_list):
            global k
            # k just stands for k-th element in listB and listG
            #assert len(listB) == len(listG)
            colorG = [0.5, 0.5, 0.5]   # gray
            color1 = [0.99, 0.0, 0.0] # red Bspline
            color2 = [0.0, 0.0, 0.99] # blue Line
            color3 = [0.0, 0.99, 0.0] # green Circle

            # arrayR and take first
            #arrayR = down_sample_point
            if k == 0:
                curves = BSpline_list
                color = color1
            elif k == 1:
                curves = Line_list
                color = color2
            elif k == 2:
                curves = Circle_list
                color = color3
            
            curves_idx = []
            for i in range(len(curves)):
                curves_idx = curves_idx + curves[i][2]
            color_array = np.zeros_like(vertices)
            color_array[curves_idx, :] = color
            if k < 2:
                k = k + 1
            point_cloud.points = open3d.utility.Vector3dVector(vertices)
            point_cloud.colors = open3d.utility.Vector3dVector(color_array)
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
            #vis.run()
        '''
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
        '''               

        curves_idx = []
        for i in range(len(BSpline_list)):
            curves_idx = curves_idx + BSpline_list[i][2]
        color_array = np.zeros_like(vertices)
        color_array[curves_idx, :] = [0.99, 0.0, 0.0] # BSplines

        # create point clouds and visualizers
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(vertices)
        point_cloud.colors = open3d.utility.Vector3dVector(color_array)

        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_callback(87, partial(update_visualization, vertices = vertices, BSpline_list = BSpline_list, Line_list = Line_list, Circle_list = Circle_list)) # W    
        '''
        vis.register_key_callback(69, partial(update_visualization32, \
                                            down_sample_point = down_sample_point, \
                                            open_gt_pair_idx = open_gt_pair_idx, \
                                            open_gt_valid_mask = open_gt_valid_mask, \
                                            open_gt_256_64_idx = open_gt_256_64_idx, \
                                            open_gt_type = open_gt_type, \
                                            open_gt_res = open_gt_res, \
                                            open_gt_sample_points = open_gt_sample_points, \
                                            open_gt_mask = open_gt_mask)) # E
        '''
        vis.register_key_callback(81, close_visualization) # Q
        vis.add_geometry(point_cloud)
        vis.run()
    vis.destroy_window()