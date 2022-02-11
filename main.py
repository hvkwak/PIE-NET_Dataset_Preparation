import os
import sys
import numpy as np
import trimesh
import scipy.io
import random
from tqdm import tqdm
from utils import delete_newline
from utils import curves_with_vertex_indices
from utils import cross_points_finder
from utils import update_lists_open
#from utils import another_half_curve_pair_exist
from utils import graipher_FPS
from utils import nearest_neighbor_finder
#from utils import greedy_nearest_neighbor_finder
from utils import log_string
#from utils import merge_two_half_circles_or_BSpline
from utils import update_lists_closed
from utils import rest_curve_finder
from utils import touch_in_circles_or_BSplines
from utils import touch_in_circle
from utils import mostly_sharp_edges
from utils import Check_Connect_Circles
from utils import part_of
from utils import check_OpenCircle
from utils import connection_available
from utils import vertex_num_finder
#from utils import Possible_Circle_in_Open_Circle
from itertools import combinations
#from utils import degrees_same
from cycle_detection import Cycle_Detector_in_BSplines
from grafkom1Framework import ObjLoader
#from utils import PathLength
#from LongestPath import Graph
#from visualizer import view_point_1
#from visualizer import view_point
import open3d
from functools import partial


if __name__ == "__main__": 

    """ generates the training dataset from ABC Dataset(Koch et al. 2019) suitable for the implementation of 
        PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020)

    Command Line Arguments:
        list_obj.txt (str): list of objects (.obj)
        list_features.txt (str): list of features (.yml)
        list_stats.txt (str) : list of stats (.yml)
    
    Returns: A Python dictionary of training dataset.
    """
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
    for i in range(1428, model_total_num):
        
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
            if vertices.shape[0] > 45000: 
                print("vertices:", vertices.shape[0], " > 40000. skip this.")
                log_string("Type 1", log_fout)
                log_string("vertices " +str(vertices.shape[0])+" > 40000. skip this.", log_fout)
                del vertices
                del faces
                del vertex_normals
                continue
            
            #Type2
            # Curves with vertex indices: (sharp and not sharp)edges of BSpline, Line, Cycle only.
            if not mostly_sharp_edges(list_ftr_line, threshold=0.30):
                print("sharp_true_count/(sharp_true_count+sharp_false_count) < 0.30. skip this.")
                log_string("Type 2", log_fout)
                log_string("sharp_true_count/(sharp_true_count+sharp_false_count) < 0.30. skip this.", log_fout)
                continue
            
            # Type3
            # This has curves other than Circle, BSpline or Line, skip this.
            all_curves = []
            try:
                all_curves = curves_with_vertex_indices(list_ftr_line, take_sharp_false = True)
            except:
                print("there are curves not in [Circle, BSpline, Line]. skip this.")
                log_string("Type 3", log_fout)
                log_string("there are curves not in [Circle, BSpline, Line]. skip this.", log_fout)
                continue                        
            curve_num = len(all_curves)

            # Preprocessing: (Re)classify / Filter points accordingly!
            BSpline_list = []
            Line_list = []
            Circle_list = []


            # Group all the curves. Object with curves with very few vertices: skip them!
            k = 0
            very_few = False
            while k < curve_num:
                curve = all_curves[k]
                if curve[0] == 'BSpline': 
                    BSpline_list.append(curve)
                    if len(curve[2]) < 3: very_few = True
                elif curve[0] == 'Line': 
                    Line_list.append(curve)
                    if len(curve[2]) < 4: very_few = True
                elif curve[0] == 'Circle': 
                    Circle_list.append(curve)
                    if len(curve[2]) < 6: very_few = True
                k = k + 1

            # skip if there is a curve which consists of very few vertices.
            if very_few:
                print("there is a curve which consists of very few vertices. skip this.")
                log_string("Type 24", log_fout)
                log_string("there is a curve which consists of very few vertices. skip this.", log_fout)
                continue
            
            
            # skip if there are one type of curves too much.
            if len(BSpline_list) > 300 or len(Circle_list) > 300 or len(Line_list) > 300:
                print("at least one curve type has > 300 curves. skip this.")
                log_string("Type 4", log_fout)
                log_string("at least one curve type has > 300 curves. skip this.", log_fout)
                continue

            # Find a misclassified BSplines and first classify them correctly into circle,
            # if they have same start and end points.
            '''
            k = 0
            BSpline_list_num = len(BSpline_list)
            while k < BSpline_list_num :
                if BSpline_list[k][2][0] == BSpline_list[k][2][-1]:
                    BSpline_list[k][0] = 'Circle'
                    Circle_list.append(BSpline_list[k])
                    del BSpline_list[k]
                    BSpline_list_num = BSpline_list_num - 1
                    k = k - 1
                k = k + 1
            '''

            # Check if there are half Circles/BSplines pair, merge them if there's one. BSplines 
            try:
                BSpline_Circle_list = rest_curve_finder(BSpline_list + Circle_list, vertices)
            except:
                print("there's something wrong in rest_curve_finder. possibly two BSplines having same start/end points, but not buliding a circle. skip this.")
                log_string("Type 25", log_fout)
                log_string("there's something wrong in rest_curve_finder. possibly two BSplines having same start/end points, but not buliding a circle. skip this.", log_fout)
                continue

            #print("len(BSpline_Circle_list): ", len(BSpline_Circle_list))
            BSpline_list = []
            Circle_list = []
            k = 0
            curve_num = len(BSpline_Circle_list)
            while k < curve_num:
                curve = BSpline_Circle_list[k]
                if curve[0] == 'BSpline': BSpline_list.append(curve)
                elif curve[0] == 'Circle': Circle_list.append(curve)
                k = k + 1
                #print(k)
            
            '''
            # check if a line touches circles_or_BSplines
            Line_list_num = len(Line_list)
            k = 0
            while k < Line_list_num:
                if touch_in_circles_or_BSplines(Line_list[k][2], Circle_list):
                    del Line_list[k]
                    k = k - 1
                    Line_list_num = Line_list_num - 1
                k = k + 1
            '''

            # Circles should be also classified into three categories: Full Circle, OpenCircle(note: it is Circle_list after this.)
            FullCircles = []
            Circle_list_num = len(Circle_list)
            k = 0
            while k < Circle_list_num:
                if Circle_list[k][2][0] == Circle_list[k][2][-1]:
                    FullCircles.append(Circle_list[k])
                    del Circle_list[k]
                    k = k - 1
                    Circle_list_num = Circle_list_num - 1
                k = k + 1
            
            '''
            # Divide them into two categories: Half Circle or BSpline.
            OpenCircle_list = []
            k = 0
            Circle_list_num = len(Circle_list)
            while k < Circle_list_num:
                if Circle_list[k][2][0] != Circle_list[k][2][-1]:
                    point1_idx = Circle_list[k][2][0]
                    point2_idx = Circle_list[k][2][-1]
                    point1 = vertices[point1_idx, :]
                    point2 = vertices[point2_idx, :]
                    middle = (point1 + point2)/2.0
                    distances = np.sqrt(((middle - vertices[Circle_list[k][2], :])**2).sum(axis = 1))
                    small_std = np.std(distances, dtype = np.float64) < 0.05

                    if len(Circle_list[k][2]) > 3 and small_std:
                        Circle_list[k][0] == 'HalfCircle'
                        HalfCircle_list.append(Circle_list[k])
                        del Circle_list[k]
                        Circle_list_num = Circle_list_num - 1
                        k = k - 1
                    else:
                        Circle_list[k][0] = 'BSpline'
                        BSpline_list.append(Circle_list[k])
                        del Circle_list[k]
                        Circle_list_num = Circle_list_num - 1
                        k = k - 1                    
                k = k + 1
            '''
            '''
            while PathLength(Circle_list, nodes) > 1:
                HalfCircle_list, Circle_list = Check_Connect_Circles(nodes, vertices, HalfCircle_list, Circle_list, BSpline_list)
            '''

            OpenCircle_list = Circle_list
            ####
            # if Circles in OpenCircle are connectable, just skip this
            '''
            if connection_available(OpenCircle_list):
                print("connection_available in Opencircle_list.")
                log_string("Type 23", log_fout)
                log_string("connection_available in Opencircle_list.", log_fout)
                continue
            '''


            # this is different to first finding out full circles. first and last vertices are not equal,
            # but there's still possibility to build circles.
            # 1. take all the open circles
            # 2. see if combinations of 1, 2, 3, .... len(OpenCircle_list) builds a circle
            '''
            OpenCircle_list_num = len(OpenCircle_list)
            sample_list = list(range(OpenCircle_list_num))
            list_combinations = list()
            for k in range(len(sample_list) + 1):
                list_combinations += list(combinations(sample_list, k))
                
            all_combinations = list_combinations[1:]
            for k in all_combinations:
                vertex_idx = []
                for i in k:
                    vertex_idx = vertex_idx + OpenCircle_list[i][2]
                
                vertex_idx = np.array([vertex_idx])
                r = np.sqrt(np.sum((np.mean(vertices[vertex_idx, ...], axis = 0) - vertices[vertex_idx, ...])**2, axis = 1))
                if np.std(r) < 0.001:
                    # this is a possible circle.
                    vertex_idx = list(vertex_idx)
                    FullCircles.append([['Circle'], None, vertex_idx])
                    indices_to_delete = k
                    sorted_indecies_to_delete = sorted(indices_to_delete, reverse=True)
                    for index in indices_to_delete:
                        del OpenCircle_list[index]
            '''
            # Check if OpenCircle_list contains something "more" than a HalfCircle (theta > pi)
            if check_OpenCircle(OpenCircle_list, vertices):
                print("OpenCircle_list contains something more than a HalfCircle (theta > pi)")
                log_string("Type 22", log_fout)
                log_string("OpenCircle_list contains something more than a HalfCircle (theta > pi)", log_fout)
                continue

            ####

            
            


            Circle_list = FullCircles
            #
            # Find BSplines with degree = 1 and classify them accordingly:
            #
            # BSpline with degree = 1 and both start/end points touch (two full)circles, remove it from the list
            # BSpline with degree = 1 and no touches -> keep them as line.
            #
            # first take all the BSplines of degree 1
            BSpline_degree_one_list = []
            BSpline_list_num = len(BSpline_list)
            k = 0
            while k < BSpline_list_num:
                if BSpline_list[k][1] == 1:
                    BSpline_degree_one_list.append(BSpline_list[k])
                    del BSpline_list[k]
                    k = k - 1
                    BSpline_list_num = BSpline_list_num - 1
                k = k + 1
            
            # Delete them or add them to Lines.
            # touching (two full)circles (or BSplines) should be eliminated.
            BSpline_degree_one_list_num = len(BSpline_degree_one_list)
            k = 0
            while k < BSpline_degree_one_list_num:
                if touch_in_circles_or_BSplines(BSpline_degree_one_list[k][2], Circle_list):
                    del BSpline_degree_one_list[k]
                    k = k - 1
                    BSpline_degree_one_list_num = BSpline_degree_one_list_num - 1
                elif not touch_in_circles_or_BSplines(BSpline_degree_one_list[k][2], Circle_list):
                    BSpline_degree_one_list[k][0] = 'Line'
                    Line_list.append(BSpline_degree_one_list[k])
                    del BSpline_degree_one_list[k]
                    k = k - 1
                    BSpline_degree_one_list_num = BSpline_degree_one_list_num - 1
                k = k + 1

            # BSplines with its degree > 1, touching two circles -> remove it
            BSpline_list_num = len(BSpline_list)
            k = 0
            while k < BSpline_list_num:
                if touch_in_circles_or_BSplines(BSpline_list[k][2], Circle_list):
                    del BSpline_list[k]
                    k = k - 1
                    BSpline_list_num = BSpline_list_num - 1
                k = k + 1

            # Lines, too. if they touch two full circles simultaneously, remove them.
            Line_list_num = len(Line_list)
            k = 0
            while k < Line_list_num:
                if touch_in_circles_or_BSplines(Line_list[k][2], Circle_list):
                    del Line_list[k]
                    k = k - 1
                    Line_list_num = Line_list_num - 1
                k = k + 1

            # Lines and BSplines: if one touch one full circle, it sounds too complicated. just skip them.
            list_num = len(Line_list) + len(BSpline_list)
            temp_list = Line_list + BSpline_list
            k = 0
            touch_in_circles_ = False
            while k < list_num:
                if touch_in_circle(temp_list[k][2], Circle_list):
                    touch_in_circles_ = True
                    break
                k = k + 1
            if touch_in_circles_:
                print("there is at least one line touching a circle. skip this.")
                log_string("Type 5", log_fout)
                log_string("there is at least one line touching a circle. skip this.", log_fout)
                continue

            # Check if multiple BSplines can form a circle.
            # first check if there are vertices that are "visited" more than twice. 
            visited_verticies = []
            BSpline_list_num = len(BSpline_list)
            
            for k in range(BSpline_list_num):
                visited_verticies.append(BSpline_list[k][2][0])
                visited_verticies.append(BSpline_list[k][2][-1])
            
            if (np.bincount(visited_verticies) > 2).sum() > 0:
                print("there exist at least one vertex that is visited more than twice. skip this.")
                log_string("Type 6", log_fout)
                log_string("there exist at least one vertex that is visited more than twice. skip this.", log_fout)
                continue
            
            # if there are at least one detected cycle in BSplines, skip it.
            print("Detecting a cycle... this can take a while....", end = " ")
            Cycle_Detector = Cycle_Detector_in_BSplines(BSpline_list)
            detected_cycles_in_BSplines = Cycle_Detector.run_cycle_detection_in_BSplines()
            result_list = Cycle_Detector.result_list
            print("Finished!")
            if detected_cycles_in_BSplines:
                print("There are at least one detected cycle, skip this.")
                log_string("Type 7", log_fout)
                log_string("there are at least one detected cycle, skip this.", log_fout)
                continue




            '''
            # run this once more to detect cycle in OpenCircle, where we try to find this specific triangular looking "opencircle" cycle.
            terminate = False
            Cycle_Detector = Cycle_Detector_in_BSplines(OpenCircle_list)
            detected_cycles_in_OpenCircles = Cycle_Detector.run_cycle_detection_in_BSplines()
            result_list = Cycle_Detector.result_list
            while detected_cycles_in_OpenCircles:
                
                if len(result_list[0]) == 3:
                    idx_to_delete = []
                    for vertex_num_to_find in result_list[0]:
                        idx_to_delete.append(vertex_num_finder(OpenCircle_list, vertex_num_to_find))
                    #idx_to_delete = result_list[0]
                    sorted_idx_to_delete = sorted(idx_to_delete, reverse=True)
                    for index in sorted_idx_to_delete:
                        del OpenCircle_list[index]                
                    Cycle_Detector = Cycle_Detector_in_BSplines(OpenCircle_list)
                    detected_cycles_in_OpenCircles = Cycle_Detector.run_cycle_detection_in_BSplines()
                
                if len(result_list[0]) > 3:
                    terminate = True
            
            if terminate:
                print("There were a cycle length > 3 in OpenCircle. skip this.")
                log_string("Type 26", log_fout)
                log_string("There were a cycle length > 3 in OpenCircle. skip this.", log_fout)
                continue
            '''

            #
            # More filtering rules for BSpline_list, Line_list, OpenCircle_list, Circle_list.
            #

            # connectable OpenCircle exists -> skip this

            # 1. Vertices of lines are completely part of BSplines or Circles -> remove them
            Line_list_num = len(Line_list)
            k = 0
            while k < len(Line_list):
                if part_of(Line_list[k][2], BSpline_list + OpenCircle_list + Circle_list):
                    del Line_list[k]
                    k = k - 1
                    Line_list_num = Line_list_num - 1
                k = k + 1

            # 2. Curves with just 1 or 2 verticies: remove them
            




            # check the visualizations.
            # create updates
            def close_visualization(vis):
                vis.close()

            s = 1 # 
            def update_visualization(vis, vertices, BSpline_list, Line_list, OpenCircle_list, Circle_list):
                global s
                print("s: ", s)
                # k just stands for k-th element in listB and listG
                #assert len(listB) == len(listG)
                #colorG = [0.5, 0.5, 0.5]   # gray
                color1 = [0.99, 0.0, 0.0] # red Bspline
                color2 = [0.0, 0.99, 0.99] # blue Line
                color3 = [0.0, 0.99, 0.0] # green Circle

                # arrayR and take first
                #arrayR = down_sample_point
                if s == 1:
                    curves = Line_list
                    color = color1 # red
                elif s == 2:
                    curves = Circle_list
                    color = color2 # lightblue
                elif s == 3:
                    curves = OpenCircle_list
                    color = color3 # green
                
                curves_idx = []
                for i in range(len(curves)):
                    curves_idx = curves_idx + curves[i][2]
                color_array = np.zeros_like(vertices)
                color_array[curves_idx, :] = color
                if s < 3:
                    s += 1
                point_cloud.points = open3d.utility.Vector3dVector(vertices)
                point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                vis.update_geometry(point_cloud)
                vis.poll_events()
                vis.update_renderer()
                #vis.run()

            curves_idx = []
            lines_idx = []
            for i in range(len(BSpline_list)):
                if k == 1 and BSpline_list[i][1] == 1:
                    lines_idx = lines_idx + BSpline_list[i][2]
                curves_idx = curves_idx + BSpline_list[i][2]
            color_array = np.zeros_like(vertices)
            color_array[curves_idx, :] = [0.0, 0.0, 0.99] # BSplines red
            if len(lines_idx) > 0:
                color_array[lines_idx, :] = [0.0, 0.99, 0.99] # exceptions
            # create point clouds and visualizers
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(vertices)
            point_cloud.colors = open3d.utility.Vector3dVector(color_array)

            vis = open3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.register_key_callback(87, partial(update_visualization, vertices = vertices, BSpline_list = BSpline_list, Line_list = Line_list, OpenCircle_list = OpenCircle_list, Circle_list = Circle_list)) # W    
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


    '''
            #
            # Classifications into open/closed curve AND edge/corner points
            #
            open_curves = []
            closed_curves = []
            corner_points_ori = []
            edge_points_ori = []
            all_curves = Line_list + Circle_list + BSpline_list
            curve_num = len(all_curves)

            k = 0
            while k < curve_num:
                curve = all_curves[k]
                # check if there are (corner) points, where two curves cross or meet. like an alphabet X
                if len(curve[2]) > 2 and k < curve_num-1:
                    for j in range(k+1, curve_num):
                        if len(all_curves[j][2]) > 2:
                            cross_points = cross_points_finder(curve[2], all_curves[j][2])
                            if len(cross_points) > 0:
                                print("len(cross_points): ", len(cross_points), "> 0.")
                                log_string("Type 8", log_fout)
                                log_string("len(cross_points) > 0.", log_fout)
                                corner_points_ori = corner_points_ori + cross_points
                k = k + 1
            del all_curves

            k = 0
            Line_Circle_List = Line_list + Circle_list
            while k < len(Line_Circle_List):
                # classifications
                curve = Line_Circle_List[k]
                if curve[0] == 'Line':
                    open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                elif curve[0] == 'Circle':
                    closed_curves, edge_points_ori = update_lists_closed(curve, closed_curves, edge_points_ori)
                k = k + 1
            del Line_Circle_List

            k = 0
            BSpline_OpenCircle_List = BSpline_list + OpenCircle_list
            while k < len(BSpline_OpenCircle_List):
                curve = BSpline_OpenCircle_List[k]
                if curve[0] == 'BSpline':
                    if 3 <= len(curve[2]) <= 6 :
                        corner_points_ori.append(curve[2][len(curve[2])//2])
                    open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                elif curve[0] == 'HalfCircle':
                    open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                k = k + 1
            del BSpline_list
            del OpenCircle_list

            # if there are more than 256 curves in each section: don't use this model.
            if (len(open_curves) > 256) or (len(closed_curves) > 256): 
                print("(open/closed)_curves > 256. skip this.")
                log_string("Type 9", log_fout)
                log_string("(open/closed)_curves > 256. skip this.", log_fout)
                continue

            # if (len(open_curves) == 0) or (len(closed_curves) == 0): 
            # just reject the object if there are no open_curves
            if (len(open_curves) == 0): 
                print("open_curves = 0. skip this.")
                log_string("Type 10", log_fout)
                log_string("open_curves = 0. skip this.", log_fout)
                continue

            # make the list unique
            edge_points_ori = np.unique(edge_points_ori)
            corner_points_ori = np.unique(corner_points_ori)
            skip_this_model = edge_points_ori.shape[0] == 0 or corner_points_ori.shape[0] == 0 \
                            or edge_points_ori.shape[0] > FPS_num or  edge_points_ori.shape[0] > FPS_num

            if skip_this_model: 
                print("problems in (edge/corner)_points_ori(.shape[0] = 0). Skip this.")
                log_string("Type 11", log_fout)
                log_string("problems in (edge/corner)_points_ori(.shape[0] = 0). Skip this.", log_fout)
                continue

            # Downsampling
            # create mesh
            mesh = trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = vertex_normals)

            # (uniform) random sample 100K surface points: Points in space on the surface of mesh
            #mesh_sample_xyz, _ = trimesh.sample.sample_surface(mesh, 100000)
            mesh_sample_xyz, _ = trimesh.sample.sample_surface_even(mesh, 100000)
            del mesh
            # (greedy) Farthest Points Sampling
            down_sample_point = graipher_FPS(mesh_sample_xyz, FPS_num) # dtype: np.float64
            del mesh_sample_xyz

            nearest_neighbor_idx_edge = 0
            nearest_neighbor_idx_corner = 0
            # Annotation transfer
            # edge_points_now ('PC_8096_edge_points_label_bin'), (8096, 1), dtype: uint8
            # Note: find a nearest neighbor of each edge_point in edge_points_ori, label it as an "edge"
            # option 1 : no clustering, just take nearest neighbors. Ties shoud be handled again with nearest neighbor
            # concept around the tie point of a down_sample_point
            try:
                nearest_neighbor_idx_edge_1 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=False, neighbor_distance=1)
                nearest_neighbor_idx_corner_1 = nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point, use_clustering=False, neighbor_distance=1)
                distance_max_1 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_1, :])**2).sum(axis = 1))
                if distance_max_1 > 1.5:
                    print("distance_max_1: ", distance_max_1, " > 1.5. skip this.")
                    log_string("Type 12", log_fout)
                    log_string("distance_max_1: "+str(distance_max_1)+ " > 1.5 skip this.", log_fout)
                    continue
                nearest_neighbor_idx_edge = nearest_neighbor_idx_edge_1
                nearest_neighbor_idx_corner = nearest_neighbor_idx_corner_1
            except:
                print("NN was not successful. skip this.")
                log_string("Type 13", log_fout)
                log_string("NN was not successful. skip this.", log_fout)
                continue
            
            if nearest_neighbor_idx_corner.shape[0] > 23:
                print("corner points > 23. skip this.")
                log_string("Type 14", log_fout)
                log_string("corner points > 23. skip this.", log_fout)
                continue



            # option 3: greedy. Just random shuffle the indicies and take distance matrix and take minimums.
            #nearest_neighbor_idx_edge_3 = greedy_nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point)
            #nearest_neighbor_idx_corner_3 = greedy_nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point)                
            #distance_mean_3 = np.mean(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_3, :])**2).sum(axis = 1))
            #distance_max_3 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_3, :])**2).sum(axis = 1))
            #log_string('Curves in the circle do not match. Skip this curve: '+str(curve), log_fout)
            #mat = scipy.io.loadmat('1.mat')

            # initialize memory arrays
            edge_points_label = np.zeros((FPS_num), dtype = np.uint8)
            corner_points_label = np.zeros((FPS_num), dtype = np.uint8)
            edge_points_residual_vector = np.zeros_like(down_sample_point)
            corner_points_residual_vector = np.zeros_like(down_sample_point)
            open_gt_pair_idx = np.zeros((256, 2), dtype=np.uint16)
            open_gt_valid_mask = np.zeros((256, 1), dtype=np.uint8)
            open_gt_256_64_idx = np.zeros((256, 64), dtype=np.uint16)
            open_gt_type = np.zeros((256, 1), dtype=np.uint8) # Note: BSpline, Lines, HalfCircle so three label types: 1, 2, 3, zero for NullClass
            open_type_onehot = np.zeros((256, 4), dtype=np.uint8)
            open_gt_res = np.zeros((256, 6), dtype=np.float32)
            open_gt_sample_points = np.zeros((256, 64, 3), dtype=np.float32)
            open_gt_mask = np.zeros((256, 64), dtype=np.uint8)
            closed_gt_256_64_idx = np.zeros((256, 64), dtype=np.uint16)
            closed_gt_mask = np.zeros((256, 64), dtype=np.uint8)
            closed_gt_type = np.zeros((256, 1), dtype=np.uint8)
            closed_gt_res = np.zeros((256, 3), dtype=np.float32)
            closed_gt_sample_points = np.zeros((256, 64, 3), dtype=np.float32)
            closed_gt_valid_mask = np.zeros((256, 1), dtype=np.uint8)
            closed_gt_pair_idx = np.zeros((256, 1), dtype=np.uint16)
            

            edge_points_label[nearest_neighbor_idx_edge] = 1
            edge_points_residual_vector[nearest_neighbor_idx_edge, :] = vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge, :]
            corner_points_label[nearest_neighbor_idx_corner] = 1
            corner_points_residual_vector[nearest_neighbor_idx_corner, ] = vertices[corner_points_ori,:] - down_sample_point[nearest_neighbor_idx_corner, :]

            # check if corner points are "safe"
            distance_between_corner_points = (np.apply_along_axis(np.subtract, 1, down_sample_point[np.where(corner_points_label == 1)[0],:], down_sample_point[np.where(corner_points_label == 1)[0],:])**2).sum(axis = 2)
            np.fill_diagonal(distance_between_corner_points, np.Inf)
            too_many_corner_points_nearby = False
            for k in range(distance_between_corner_points.shape[0]):
                # check if 10% of the all corner points gathered in a neighborhood within the distance of 5.
                if (distance_between_corner_points[k,:] < 5.0).sum() / distance_between_corner_points.shape[0] > 0.1:
                    too_many_corner_points_nearby = True
                    break
            if too_many_corner_points_nearby:
                print("too_many_corner_points_nearby. skip this.")
                log_string("Type 15", log_fout)
                log_string("too_many_corner_points_nearby. skip this.", log_fout)
                continue
                
            # normalize them to keep all in [-0.5, 0.5]
            max_x_in_this_model = np.max([np.max(down_sample_point[:, 0]), np.abs(np.min(down_sample_point[:, 0]))])
            max_y_in_this_model = np.max([np.max(down_sample_point[:, 1]), np.abs(np.min(down_sample_point[:, 1]))])
            max_z_in_this_model = np.max([np.max(down_sample_point[:, 2]), np.abs(np.min(down_sample_point[:, 2]))])
            max_in_this_model = np.max([max_x_in_this_model, max_y_in_this_model, max_z_in_this_model])

            if max_in_this_model > 0.5:
                down_sample_point[:,0] = (down_sample_point[:, 0] / (max_in_this_model*2.0))
                down_sample_point[:,1] = (down_sample_point[:, 1] / (max_in_this_model*2.0))
                down_sample_point[:,2] = (down_sample_point[:, 2] / (max_in_this_model*2.0))
                edge_points_residual_vector[:,0] = (edge_points_residual_vector[:, 0] / (max_in_this_model*2.0))
                edge_points_residual_vector[:,1] = (edge_points_residual_vector[:, 1] / (max_in_this_model*2.0))
                edge_points_residual_vector[:,2] = (edge_points_residual_vector[:, 2] / (max_in_this_model*2.0))
                corner_points_residual_vector[:,0] = (corner_points_residual_vector[:, 0] / (max_in_this_model*2.0))
                corner_points_residual_vector[:,1] = (corner_points_residual_vector[:, 1] / (max_in_this_model*2.0))
                corner_points_residual_vector[:,2] = (corner_points_residual_vector[:, 2] / (max_in_this_model*2.0))

            m = 0
            closed_curve_NN_search_failed = False
            down_sample_point_copy = down_sample_point.copy()
            for curve in closed_curves:
                # first element
                try:
                    closed_gt_pair_idx[m,0] = nearest_neighbor_finder(vertices[np.array([curve[2][0]]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    closed_gt_256_64_idx[m, 0] = closed_gt_pair_idx[m,0]
                    # make the point unavailable.
                    down_sample_point_copy[closed_gt_pair_idx[m,0], :] = np.Inf
                except:
                    print("NN for closed_gt_pair_idx was not successful. skip this.")
                    log_string("Type 16", log_fout)
                    log_string("NN for closed_gt_pair_idx was not successful. skip this.", log_fout)
                    closed_curve_NN_search_failed = True
                closed_gt_valid_mask[m, 0] = 1
                
                if curve[2][0] == curve[2][-1]: curve[2] = curve[2][:-1] # update if these two indicies are same.

                # the rest of them!
                if len(curve[2]) > 64:
                    # take start/end points + sample 62 points = 64 points
                    try:
                        closed_gt_256_64_idx[m, 1:64] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:63]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        # make the point unavailable.
                        down_sample_point_copy[closed_gt_256_64_idx[m, 1:64], :] = np.Inf
                    except:
                        print("NN for closed_gt_256_64_idx was not successful. skip this.")
                        log_string("Type 17", log_fout)
                        log_string("NN for closed_gt_256_64_idx was not successful. skip this.", log_fout)
                        closed_curve_NN_search_failed = True
                        continue
                    #closed_gt_256_64_idx[i, 63] = curve[2][-1]
                    closed_gt_mask[m, 0:64] = 1
                else:
                    try:
                        closed_gt_256_64_idx[m, 1:len(curve[2][1:])+1] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:len(curve[2][1:])]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        # make the point unavailable.
                        down_sample_point_copy[closed_gt_256_64_idx[m, 1:len(curve[2][1:])+1], :] = np.Inf
                    except:
                        print("NN for closed_gt_256_64_idx len() < 64 was not successful. skip this.")
                        log_string("Type 18", log_fout)
                        log_string("NN for closed_gt_256_64_idx len() < 64 was not successful. skip this.", log_fout)
                        closed_curve_NN_search_failed = True
                        continue
                    closed_gt_256_64_idx[m, len(curve[2][1:])+1:] = closed_gt_256_64_idx[m, len(curve[2][1:])]
                    closed_gt_mask[m, :len(curve[2])] = 1
                
                
                # closed_gt_type, closed_type_onehot
                if curve[0] == "Circle": closed_gt_type[m,0] = 1
                
                # closed_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[closed_gt_pair_idx[m,][0], ]
                closed_gt_res[m, ] = np.array([res1])

                # open_gt_sample_points
                closed_gt_sample_points[m, ...] = down_sample_point[closed_gt_256_64_idx[m], ]
                m = m + 1
            if closed_curve_NN_search_failed: continue

            n = 0
            open_curve_NN_search_failed = False
            down_sample_point_copy = down_sample_point.copy()
            for curve in open_curves:
                # first and last element
                try:
                    open_gt_pair_idx[n,] = nearest_neighbor_finder(vertices[np.array([curve[2][0], curve[2][-1]]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    open_gt_256_64_idx[n, 0] = open_gt_pair_idx[n,0]
                    open_gt_valid_mask[n, 0] = 1
                    down_sample_point_copy[open_gt_pair_idx[n,0], :] = np.Inf
                    down_sample_point_copy[open_gt_pair_idx[n,1], :] = np.Inf
                except:
                    print("NN for open_gt_pair_idx was not successful. skip this.")
                    log_string("Type 19", log_fout)
                    log_string("NN for open_gt_pair_idx was not successful. skip this.", log_fout)
                    open_curve_NN_search_failed = True
                    continue
                
                # open_gt_256_64_idx
                if len(curve[2]) > 64:
                    # sample start/end points + sample 62 points = 64 points
                    try:
                        open_gt_256_64_idx[n, 1:63] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))[:62]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        down_sample_point_copy[open_gt_256_64_idx[n, 1:63],:] = np.Inf
                    except:
                        print("NN for open_gt_256_64_idx was not successful. skip this.")
                        log_string("Type 20", log_fout)
                        log_string("NN for open_gt_256_64_idx was not successful. skip this.", log_fout)
                        open_curve_NN_search_failed = True
                        continue
                    open_gt_256_64_idx[n, 63] = open_gt_pair_idx[n,1]
                    open_gt_mask[n, 0:64] = 1
                else:
                    middle_idx_num = len(curve[2]) - 2
                    #open_gt_256_64_idx[n, :] = curve[2] + [curve[2][-1]]*(64 - indicies_num)
                    try:
                        open_gt_256_64_idx[n, 1:(middle_idx_num+1)] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        down_sample_point_copy[open_gt_256_64_idx[n, 1:(middle_idx_num+1)],:] = np.Inf
                    except:
                        print("NN for open_gt_256_64_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.")
                        log_string("Type 21", log_fout)
                        log_string("NN for open_gt_256_64_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.", log_fout)
                        open_curve_NN_search_failed = True
                        continue
                    open_gt_256_64_idx[n, (middle_idx_num+1):64] = open_gt_pair_idx[n, 1]
                    open_gt_mask[n, 0:(middle_idx_num+2)] = 1

                # open_gt_type, open_type_onehot BSpline, Lines, HalfCircle
                if curve[0] == 'BSpline': open_gt_type[n,0], open_type_onehot[n, ] = 1, np.array([0, 1, 0, 0])
                elif curve[0] == 'Line' : open_gt_type[n,0], open_type_onehot[n, ] = 2, np.array([0, 0, 1, 0]) # "Line"
                elif curve[0] == 'HalfCircle' : open_gt_type[n,0], open_type_onehot[n, ] = 3, np.array([0, 0, 0, 1]) # "HalfCircle"
                
                # open_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[open_gt_pair_idx[n, ][0], ]
                res2 = vertices[curve[2][-1], ]-down_sample_point[open_gt_pair_idx[n, ][1], ]
                open_gt_res[n, ] = np.array([res1, res2]).flatten()

                # open_gt_sample_points
                open_gt_sample_points[n, ...] = down_sample_point[open_gt_256_64_idx[n], ]
                n = n + 1
            if open_curve_NN_search_failed: continue
            
            print("Ok. save data.")
            log_string("Ok. save data.", log_fout)
            #view_point_1(down_sample_point, np.where(edge_points_label == 1)[0], np.where(corner_points_label == 1)[0])
            tp = np.dtype([
                ('down_sample_point', 'O'),
                ('edge_points_label', 'O'),
                ('edge_points_residual_vector', 'O'),
                ('corner_points_label', 'O'),
                ('corner_points_residual_vector', 'O'),
                  ('open_gt_pair_idx', 'O'),
                ('closed_gt_pair_idx', 'O'),
                  ('open_gt_valid_mask', 'O'),
                ('closed_gt_valid_mask', 'O'),
                  ('open_gt_256_64_idx', 'O'),
                ('closed_gt_256_64_idx', 'O'),
                  ('open_gt_type','O'),
                ('closed_gt_type','O'),
                ('open_type_onehot','O'),
                  ('open_gt_res', 'O'),
                ('closed_gt_res', 'O'),
                  ('open_gt_sample_points', 'O'),
                ('closed_gt_sample_points', 'O'), 
                  ('open_gt_mask', 'O'),
                ('closed_gt_mask', 'O')
                ])
            data['Training_data'][batch_count, 0] = np.zeros((1, 1), dtype = tp)
            for tp_name in tp.names: 
                save_this = locals()[tp_name]
                data['Training_data'][batch_count, 0][tp_name][0, 0] = save_this
            
        if batch_count == 63:
            file_ = save_prefix+"_"+str(file_count)+".mat"
            data['batch_count'] = batch_count+1
            scipy.io.savemat(file_, data)
            batch_count = 0
            file_count = file_count + 1
            data = {'batch_count': batch_count, 'Training_data': np.zeros((64, 1), dtype = object)}
        else:
            batch_count = batch_count + 1

        list_obj_line = list_obj_file.readline()
        list_ftr_line = list_ftr_file.readline()

    if batch_count > 0:
        file_ = save_prefix+"_"+str(file_count)+"_end"+".mat"
        data['batch_count'] = batch_count
        scipy.io.savemat(file_, data)
    '''

    list_obj_file.close()
    list_ftr_file.close()
    log_fout.close()



