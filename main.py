import math
import os
import sys
import numpy as np
import trimesh
import scipy.io
import random
from tqdm import tqdm
from utils import delete_newline, touching_four_BO_Line, touching_four_BO_BO
from utils import curves_with_vertex_indices
from utils import cross_points_finder
from utils import update_lists_open
#from utils import another_half_curve_pair_exist
from utils import graipher_FPS
from utils import graipher_FPS_idx_collector
from utils import nearest_neighbor_finder
#from utils import greedy_nearest_neighbor_finder
from utils import log_string
#from utils import merge_two_half_circles_or_BSpline
from utils import update_lists_closed
from utils import rest_curve_finder
from utils import touch_in_circles_or_BSplines
from utils import touch_in_circle
#from utils import mostly_sharp_edges
#from utils import Check_Connect_Circles
from utils import part_of
from utils import check_OpenCircle
#import open3d

#from utils import connection_available
#from utils import vertex_num_finder
#from utils import Possible_Circle_in_Open_Circle
#from itertools import combinations
#from utils import degrees_sameddd
from cycle_detection import Cycle_Detector_in_BSplines
from grafkom1Framework import ObjLoader

#from utils import PathLength
#from LongestPath import Graph
#from visualizer import view_point_1
#from visualizer import view_point
#import open3d
#from functools import partial


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
    
    # control visualization points: Enter '4' if you wish no visualizations
    check_point1 = args[4] == '1'
    check_point2 = args[4] == '2'
    check_point3 = args[4] == '3'
    sn = int(args[5]) # subsampling number. let us try 64 and 128.
    #plus_rate = float(args[5])*5.0*0.01
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

            subsample_rate = (FPS_num*1.0)/vertices.shape[0]
            # apply emp. discovered subsample rates:
            subsample_rate = subsample_rate + 0.15

            # Type1
            # make sure we have < 30K vertices to keep it simple.
            if vertices.shape[0] > 30000: 
                print("vertices:", vertices.shape[0], " > 30000. skip this.")
                log_string("Type 1", log_fout)
                log_string("vertices " +str(vertices.shape[0])+" > 30000. skip this.", log_fout)
                del vertices
                del faces
                del vertex_normals
                continue
            elif vertices.shape[0] < 10000:
                print("vertices:", vertices.shape[0], " < 10000. skip this.")
                log_string("Type 2", log_fout)
                log_string("vertices " +str(vertices.shape[0])+" < 10000. skip this.", log_fout)
                del vertices
                del faces
                del vertex_normals
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
                log_string("Type 4", log_fout)
                log_string("there is a curve which consists of very few vertices. skip this.", log_fout)
                continue
            
            
            # skip if there are one type of curves too much.
            if len(BSpline_list) > 300 or len(Circle_list) > 300 or len(Line_list) > 300:
                print("at least one curve type has > 300 curves. skip this.")
                log_string("Type 5", log_fout)
                log_string("at least one curve type has > 300 curves. skip this.", log_fout)
                continue


            # Check if there are half Circles/BSplines pair, merge them if there's one. BSplines 
            try:
                BSpline_Circle_list = rest_curve_finder(BSpline_list + Circle_list, vertices)
            except:
                print("there's something wrong in rest_curve_finder. possibly two BSplines having same start/end points, but not buliding a circle. skip this.")
                log_string("Type 6", log_fout)
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

            if len(FullCircles) > 0:
                print("there's at least one full circles. we don't need this. skip this.")
                log_string("Type 7", log_fout)
                log_string("there's at least one full circles. we don't need this. skip this.", log_fout)
                continue

            OpenCircle_list = Circle_list

            # Check if OpenCircle_list contains something "more" than a HalfCircle (theta > pi)
            if check_OpenCircle(OpenCircle_list, vertices):
                print("OpenCircle_list contains something more than a HalfCircle (theta > pi)")
                log_string("Type 8", log_fout)
                log_string("OpenCircle_list contains something more than a HalfCircle (theta > pi)", log_fout)
                continue

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
                log_string("Type 9", log_fout)
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
                log_string("Type 10", log_fout)
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
                log_string("Type 11", log_fout)
                log_string("there are at least one detected cycle, skip this.", log_fout)
                continue

            #
            # More filtering rules for BSpline_list, Line_list, OpenCircle_list, Circle_list.
            #

            # 1. Vertices of lines are completely part of BSplines or Circles -> remove them
            Line_list_num = len(Line_list)
            k = 0
            while k < len(Line_list):
                if part_of(Line_list[k][2], BSpline_list + OpenCircle_list + Circle_list):
                    del Line_list[k]
                    k = k - 1
                    Line_list_num = Line_list_num - 1
                k = k + 1

            #
            # Classifications into open/closed curve AND edge/corner points
            #
            open_curves = []
            closed_curves = []
            corner_points_ori = []
            edge_points_ori = []
            all_curves = Line_list + Circle_list + BSpline_list
            curve_num = len(all_curves)

            # if this object consists of only lines, this can be dropped out, take only 5%
            skip_this_model = False
            if len(Circle_list) == 0 and len(BSpline_list) == 0 and len(OpenCircle_list) == 0:
                dropout_num = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 1)
                skip_this_model = dropout_num[0] > 0

            if skip_this_model: 
                print("This object is dropped out. Skip this.")
                log_string("Type 12", log_fout)
                log_string("This object is dropped out. Skip this.", log_fout)
                continue

            # it's not likley, but let's do this: cross points check
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
                                log_string("Type 13", log_fout)
                                log_string("len(cross_points) > 0.", log_fout)
                                corner_points_ori = corner_points_ori + cross_points
                k = k + 1
            del all_curves

            # There are specific lines touching 4(2 lines + 2 curves) like this. This should be removed.
            # 
            # )_)
            # | |
            # 
            BSpline_OpenCircle_List = BSpline_list + OpenCircle_list
            before_num = len(Line_list)
            #Line_list_copy = Line_list[:]
            delete_idx_collector = []
            skip_this_model = False
            #print("BEFORE len(Line_list): ", len(Line_list))
            k = 0            
            while k < len(Line_list) and not skip_this_model:
                try:
                    if touching_four_BO_Line(BSpline_OpenCircle_List, Line_list, k, vertices) or touching_four_BO_BO(BSpline_OpenCircle_List, Line_list, k, vertices):
                        delete_idx_collector.append(k)
                        #original_collector.append(original_idx)
                    k = k + 1
                    #original_idx = original_idx + 1
                except:
                    skip_this_model = True   
                    k = k + 1
                    
            if skip_this_model:
                print("there's something wrong in touching_four_BO_Line() or touching_four_BO_BO(). skip this.")
                log_string("Type 14", log_fout)
                log_string("there's something wrong in touching_four_BO_Llsine() or touching_four_BO_BO(). skip this.", log_fout)
                continue

            for index in sorted(delete_idx_collector, reverse = True):
                del Line_list[index]            


            # corners not clear: BSplines meeting BSplines
            k = 0
            theta_threshold_1 = 2.094
            theta_threshold_2 = 3.15
            skip_this_model = False
            while k < len(BSpline_OpenCircle_List) and not skip_this_model:
                for i in range(k, len(BSpline_OpenCircle_List)):
                    current_idx = BSpline_OpenCircle_List[k][2]
                    if current_idx[0] == BSpline_OpenCircle_List[i][2][0]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)                        
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[0] == BSpline_OpenCircle_List[i][2][-1]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == BSpline_OpenCircle_List[i][2][0]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)                        
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == BSpline_OpenCircle_List[i][2][-1]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                k += 1

            if skip_this_model:
                print("Problemkreis begrenzen! skip this.")
                log_string("Type 15", log_fout)
                log_string("Problemkreis begrenzen! skip this.", log_fout)
                continue

            # corners not clear: Line meeting Line
            k = 0
            theta_threshold_1 = 2.094
            theta_threshold_2 = 3.15
            skip_this_model = False
            while k < len(Line_list) and not skip_this_model:
                for i in range(k, len(Line_list)):
                    current_idx = Line_list[k][2]
                    if current_idx[0] == Line_list[i][2][0]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = Line_list[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[0] == Line_list[i][2][-1]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = Line_list[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == Line_list[i][2][0]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = Line_list[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == Line_list[i][2][-1]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = Line_list[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                k += 1

            if skip_this_model:
                print("Problemkreis begrenzen! skip this.")
                log_string("Type 17", log_fout)
                log_string("Problemkreis begrenzen! skip this.", log_fout)
                continue

            # corners not clear? skip this object: Lines meeting BSplines
            k = 0
            skip_this_model = False
            while k < len(Line_list) and not skip_this_model:
                for i in range(len(BSpline_OpenCircle_List)):
                    current_idx = Line_list[k][2]
                    if current_idx[0] == BSpline_OpenCircle_List[i][2][0]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[0] == BSpline_OpenCircle_List[i][2][-1]:
                        corner_idx0 = current_idx[0]
                        corner_idx1 = current_idx[1]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == BSpline_OpenCircle_List[i][2][0]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][1]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                    elif current_idx[-1] == BSpline_OpenCircle_List[i][2][-1]:
                        corner_idx0 = current_idx[-1]
                        corner_idx1 = current_idx[-2]
                        corner_idx2 = BSpline_OpenCircle_List[i][2][-2]
                        vec1 = vertices[corner_idx1, ...] - vertices[corner_idx0, ...]
                        vec2 = vertices[corner_idx2, ...] - vertices[corner_idx0, ...]
                        arccos_arg = np.sum(vec1*vec2)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2))).astype(np.float64)
                        if np.abs(arccos_arg - 0.0001) < 0.001:
                            arccos_arg = 0.99
                        theta = np.arccos(arccos_arg)
                        if theta_threshold_1 < theta < theta_threshold_2:
                            skip_this_model = True
                            break
                k = k + 1

            if skip_this_model:
                print("Problemkreis begrenzen! skip this.")
                log_string("Type 16", log_fout)
                log_string("Problemkreis begrenzen! skip this.", log_fout)
                continue


            #if len(Line_list) - before_num < 0:
            if False:
                # create updates
                def close_visualization(vis):
                    vis.close()
                '''
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
                '''
                color_array = np.zeros_like(vertices)
                k = 0
                while k < len(BSpline_OpenCircle_List):
                    color_array[BSpline_OpenCircle_List[k][2], ...] = [0.0, 0.0, 0.99]
                    k = k + 1
                
                k = 0
                while k < len(Line_list_copy):
                    if k in delete_idx_collector:
                        color_array[Line_list_copy[k][2], ...] = [0.0, 0.99, 0.0]
                    else:
                        color_array[Line_list_copy[k][2], ...] = [0.0, 0.0, 0.99]
                    k = k + 1

                # create point clouds and visualizers
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(vertices)
                point_cloud.colors = open3d.utility.Vector3dVector(color_array)

                vis = open3d.visualization.VisualizerWithKeyCallback()
                vis.create_window()
                #vis.register_key_callback(87, partial(update_visualization, vertices = vertices, BSpline_list = BSpline_list, Line_list = Line_list, OpenCircle_list = OpenCircle_list, Circle_list = Circle_list)) # W    
                
                #vis.register_key_callback(69, partial(update_visualization32, \
                #                                    down_sample_point = down_sample_point, \
                #                                    open_gt_pair_idx = open_gt_pair_idx, \
                #                                    open_gt_valid_mask = open_gt_valid_mask, \
                #                                    open_gt_256_sn_idx = open_gt_256_sn_idx, \
                #                                    open_gt_type = open_gt_type, \
                #                                    open_gt_res = open_gt_res, \
                #                                    open_gt_sample_points = open_gt_sample_points, \
                #                                    open_gt_mask = open_gt_mask)) # E
                
                vis.register_key_callback(81, close_visualization) # Q
                vis.add_geometry(point_cloud)
                vis.run()
            


            # Classify lines and (full)circles
            k = 0
            Line_Circle_List = Line_list + Circle_list
            skip_this_model = False
            while k < len(Line_Circle_List) and not skip_this_model:
                curve = Line_Circle_List[k]
                sample_num = math.ceil(len(curve[2])*subsample_rate)
                # down size the curve. We are downsizing to FPS_num. curves need to be downsized accordingly. 
                # note: 8096/40000 ~ 0.20. We take first and last point, reduce rest of them according to rate dynamically..
                #
                '''
                if 3 <= len(curve[2]) <= 4:
                    curve[2] = [curve[2][0]] + random.sample(curve[2][1:-1], 1) + [curve[2][-1]]
                elif 4 < len(curve[2]) <= 6:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), 2)
                    new_end = stacked_col[0, 1]
                    stacked_col[0, 1] = stacked_col[0, -1]
                    stacked_col[0, -1] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                elif 6 < len(curve[2]) < 10:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), 3)
                    new_end = stacked_col[0, 1]
                    stacked_col[0, 1] = stacked_col[0, -1]
                    stacked_col[0, -1] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                '''
                if sample_num < 6:
                    skip_this_model = True
                    break
                else:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), sample_num)
                    new_end = stacked_col[1, 0]
                    stacked_col[1, 0] = stacked_col[-1, 0]
                    stacked_col[-1, 0] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                    #curve[2] = [curve[2][0]] + random.sample(curve[2][1:-1], sample_num) + [curve[2][-1]]

                # update lists
                if curve[0] == 'Line':
                    open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                elif curve[0] == 'Circle':
                    closed_curves, edge_points_ori = update_lists_closed(curve, closed_curves, edge_points_ori)
                k = k + 1
            del Line_Circle_List

            if skip_this_model:
                print("subsampled curve too short. skip this.")
                log_string("Type 18", log_fout)
                log_string("subsampled curve too short. skip this.", log_fout)                
                continue

            # Merge Bsplines and OpenCircles. don't forget to change the name of curve first!
            k = 0
            while k < len(OpenCircle_list):
                # classifications
                curve = OpenCircle_list[k]
                if curve[0] == 'Circle':
                    curve[0] = 'BSpline'
                k = k + 1

            k = 0
            BSpline_OpenCircle_List = BSpline_list + OpenCircle_list
            while k < len(BSpline_OpenCircle_List):
                curve = BSpline_OpenCircle_List[k]
                sample_num = math.ceil(len(curve[2])*subsample_rate)
                '''
                if 3 <= len(curve[2]) <= 4:
                    curve[2] = [curve[2][0]] + random.sample(curve[2][1:-1], 1) + [curve[2][-1]]
                elif 4 < len(curve[2]) <= 6:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), 2)
                    new_end = stacked_col[0, 1]
                    stacked_col[0, 1] = stacked_col[0, -1]
                    stacked_col[0, -1] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                elif 6 < len(curve[2]) < 10:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), 3)
                    new_end = stacked_col[0, 1]
                    stacked_col[0, 1] = stacked_col[0, -1]
                    stacked_col[0, -1] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                '''
                if sample_num < 6:
                    skip_this_model = True
                    break
                else:
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), sample_num)
                    new_end = stacked_col[1, 0]
                    stacked_col[1, 0] = stacked_col[-1, 0]
                    stacked_col[-1, 0] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                    #curve[2] = [curve[2][0]] + random.sample(curve[2][1:-1], sample_num) + [curve[2][-1]]

                    #rate = min(float(FPS_num)/vertices.shape[0] + plus_rate, 0.9)
                    #sample_num = round(len(curve[2][1:-1])*rate) - 1
                    #sample_num = len(curve[2][1:-1])
                    #curve[2] = [curve[2][0]] + random.sample(curve[2][1:-1], sample_num) + [curve[2][-1]]

                open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                k = k + 1
            del BSpline_list
            del OpenCircle_list

            if skip_this_model:
                print("subsampled curve too short. skip this.")
                log_string("Type 19", log_fout)
                log_string("subsampled curve too short. skip this.", log_fout)                
                continue

            # if there are more than 256 curves in each section: don't use this model.
            if (len(open_curves) > 256) or (len(closed_curves) > 256): 
                print("(open/closed)_curves > 256. skip this.")
                log_string("Type 20", log_fout)
                log_string("(open/closed)_curves > 256. skip this.", log_fout)
                continue

            # if (len(open_curves) == 0) or (len(closed_curves) == 0): 
            # just reject the object if there are no open_curves: this will exclude objects with circles only.
            if (len(open_curves) == 0): 
                print("open_curves = 0. skip this.")
                log_string("Type 21", log_fout)
                log_string("open_curves = 0. skip this.", log_fout)
                continue

            # make the list unique
            edge_points_ori = np.unique(edge_points_ori)
            corner_points_ori = np.unique(corner_points_ori)
            skip_this_model = edge_points_ori.shape[0] == 0 or corner_points_ori.shape[0] == 0 \
                            or edge_points_ori.shape[0] > FPS_num or  edge_points_ori.shape[0] > FPS_num \
                                or corner_points_ori.shape[0] < 5

            if skip_this_model: 
                print("problems in (edge/corner)_points_ori(.shape[0] = 0). Skip this.")
                log_string("Type 22", log_fout)
                log_string("problems in (edge/corner)_points_ori(.shape[0] = 0). Skip this.", log_fout)
                continue
            ''' 
            if check_point2:
                # create updates
                def close_visualization(vis):
                    vis.close()

                s = 1 # 
                def update_visualization_open_curve_ori(vis, vertices, open_curves, edge_points_ori, corner_points_ori):
                    global s
                    print("visualization:", s, "/", len(open_curves))

                    color_array = np.zeros_like(vertices)
                    color_array[edge_points_ori, :] = [0.0, 0.0, 0.99] # red
                    color_array[corner_points_ori, :] = [0.99, 0.0, 0.0] # blue
                    color_array[open_curves[s][2][1:-1], :] = [0.0, 0.99, 0.0]

                    if s < len(open_curves)-1:
                        s += 1
                    point_cloud.points = open3d.utility.Vector3dVector(vertices)
                    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()
                    #vis.run()

                a = 0 # 
                def update_visualization_closed_curve_ori(vis, vertices, closed_curves, edge_points_ori):
                    global a
                    print(a, "th visualization of ", len(closed_curves))

                    color_array = np.zeros_like(vertices)
                    color_array[edge_points_ori, :] = [0.0, 0.0, 0.99] # red
                    color_array[closed_curves[a][2], :] = [0.0, 0.99, 0.0]
                    #color_array[corner_points_ori, :] = [0.99, 0.0, 0.0] # blue

                    if a < len(closed_curves)-1:
                        a += 1
                    point_cloud.points = open3d.utility.Vector3dVector(vertices)
                    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()
                    #vis.run()                

                color_array = np.zeros_like(vertices)                
                color_array[edge_points_ori, :] = [0.0, 0.0, 0.99] # red
                color_array[corner_points_ori, :] = [0.99, 0.0, 0.0] # blue
                color_array[open_curves[0][2][1:-1], :] = [0.0, 0.99, 0.0]
                
                # create point clouds and visualizers
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(vertices)
                point_cloud.colors = open3d.utility.Vector3dVector(color_array)

                vis = open3d.visualization.VisualizerWithKeyCallback()
                vis.create_window()
                vis.register_key_callback(87, partial(update_visualization_open_curve_ori, vertices = vertices, open_curves = open_curves, edge_points_ori = edge_points_ori, corner_points_ori = corner_points_ori)) # W    
                vis.register_key_callback(69, partial(update_visualization_closed_curve_ori, vertices = vertices, closed_curves = closed_curves, edge_points_ori = edge_points_ori)) # E
                #vis.register_key_callback(69, partial(update_visualization32, \
                #                                    down_sample_point = down_sample_point, \
                #                                    open_gt_pair_idx = open_gt_pair_idx, \
                #                                    open_gt_valid_mask = open_gt_valid_mask, \
                #                                    open_gt_256_sn_idx = open_gt_256_sn_idx, \
                #                                    open_gt_type = open_gt_type, \
                #                                    open_gt_res = open_gt_res, \
                #                                    open_gt_sample_points = open_gt_sample_points, \
                #                                    open_gt_mask = open_gt_mask)) # E
                
                vis.register_key_callback(81, close_visualization) # Q
                vis.add_geometry(point_cloud)
                vis.run()
            '''            
            # checking the number of corner points will save us a lot of time.
            if corner_points_ori.shape[0] > 23:
                print("corner points > 23. skip this.")
                log_string("Type 23", log_fout)
                log_string("corner points > 23. skip this.", log_fout)
                continue


            # normalize vertices to keep all in [-0.5, 0.5] 
            x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
            y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
            z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
            m_x = (x_min+x_max)/2.0
            m_y = (y_min+y_max)/2.0
            m_z = (z_min+z_max)/2.0
            vertices[:, 0] = vertices[:, 0] - m_x
            vertices[:, 1] = vertices[:, 1] - m_y
            vertices[:, 2] = vertices[:, 2] - m_z
            xyz_max = np.max([x_max-x_min, y_max-y_min, z_max-z_min])
            vertices[:, 0] = vertices[:, 0]/xyz_max
            vertices[:, 1] = vertices[:, 1]/xyz_max
            vertices[:, 2] = vertices[:, 2]/xyz_max

            # check if corners are too close:
            temp_mat = np.sqrt(np.sum((np.expand_dims(vertices[corner_points_ori, ...], axis = 1) - vertices[corner_points_ori, ...])**2, axis = 2))
            np.fill_diagonal(temp_mat, np.Inf)
            if temp_mat.min() < 0.09:
                print("corner points too close < 0.09 skip this.")
                log_string("Type 23.5", log_fout)
                log_string("corner points too close < 0.09 skip this.", log_fout)
                continue

            
            
            # Downsampling
            # create mesh - this takes a lot of time.
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
                distance_max_1 = np.max(np.sqrt(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_1, :])**2).sum(axis = 1)))
                log_string(str(distance_max_1), log_fout)
                threshold = 0.03
                if distance_max_1 > threshold:
                    print("distance_max_1: ", distance_max_1, " >", threshold, ". skip this.")
                    log_string("Type 24", log_fout)
                    log_string("distance_max_1: "+str(distance_max_1)+ " >", threshold, ". skip this.", log_fout)
                    continue
                nearest_neighbor_idx_edge = nearest_neighbor_idx_edge_1
                nearest_neighbor_idx_corner = nearest_neighbor_idx_corner_1
            except:
                print("NN was not successful. skip this.")
                log_string("Type 25", log_fout)
                log_string("NN was not successful. skip this.", log_fout)
                continue
            



            # initialize memory arrays
            edge_points_label = np.zeros((FPS_num), dtype = np.uint8)
            corner_points_label = np.zeros((FPS_num), dtype = np.uint8)
            edge_points_residual_vector = np.zeros_like(down_sample_point)
            corner_points_residual_vector = np.zeros_like(down_sample_point)
            open_gt_pair_idx = np.zeros((256, 2), dtype=np.uint16)
            open_gt_valid_mask = np.zeros((256, 1), dtype=np.uint8)
            open_gt_256_sn_idx = np.zeros((256, sn), dtype=np.uint16)
            open_gt_type = np.zeros((256, 1), dtype=np.uint8) # Note: BSpline, Lines and Null
            open_type_onehot = np.zeros((256, 3), dtype=np.uint8)
            open_gt_res = np.zeros((256, 6), dtype=np.float32)
            open_gt_sample_points = np.zeros((256, sn, 3), dtype=np.float32)
            open_gt_mask = np.zeros((256, sn), dtype=np.uint8)
            closed_gt_256_sn_idx = np.zeros((256, sn), dtype=np.uint16)
            closed_gt_mask = np.zeros((256, sn), dtype=np.uint8)
            closed_gt_type = np.zeros((256, 1), dtype=np.uint8)
            closed_gt_res = np.zeros((256, 3), dtype=np.float32)
            closed_gt_sample_points = np.zeros((256, sn, 3), dtype=np.float32)
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
                if (distance_between_corner_points[k,:] < 0.02).sum() / distance_between_corner_points.shape[0] > 0.1:
                    too_many_corner_points_nearby = True
                    break
            if too_many_corner_points_nearby:
                print("too_many_corner_points_nearby. skip this.")
                log_string("Type 26", log_fout)
                log_string("too_many_corner_points_nearby."+str(distance_between_corner_points[k,:])+"skip this.", log_fout)
                continue


            m = 0
            closed_curve_NN_search_failed = False
            for curve in closed_curves:
                down_sample_point_copy = down_sample_point.copy()

                if len(curve[2]) > sn:
                    # we can only take sn points for GT!
                    sample_num = sn
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), sample_num)
                    new_end = stacked_col[1, 0]
                    stacked_col[1, 0] = stacked_col[-1, 0]
                    stacked_col[-1, 0] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))

                # first element
                try:
                    closed_gt_pair_idx[m,0] = nearest_neighbor_finder(vertices[np.array([curve[2][0]]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    closed_gt_256_sn_idx[m, 0] = closed_gt_pair_idx[m,0]
                    down_sample_point_copy[closed_gt_pair_idx[m,0], :] = np.Inf
                except:
                    print("NN for closed_gt_pair_idx was not successful. skip this.")
                    log_string("Type 27", log_fout)
                    log_string("NN for closed_gt_pair_idx was not successful. skip this.", log_fout)
                    closed_curve_NN_search_failed = True
                    break
                closed_gt_valid_mask[m, 0] = 1
                
                if curve[2][0] == curve[2][-1]: curve[2] = curve[2][:-1] # update if these two indicies are same.

                try:
                    closed_gt_256_sn_idx[m, 1:len(curve[2][1:])+1] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:len(curve[2][1:])]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    # make the point unavailable.
                    down_sample_point_copy[closed_gt_256_sn_idx[m, 1:len(curve[2][1:])+1], :] = np.Inf
                except:
                    print("NN for closed_gt_256_sn_idx len() < sn was not successful. skip this.")
                    log_string("Type 28", log_fout)
                    log_string("NN for closed_gt_256_sn_idx len() < sn was not successful. skip this.", log_fout)
                    closed_curve_NN_search_failed = True
                    break
                closed_gt_256_sn_idx[m, len(curve[2][1:])+1:] = closed_gt_256_sn_idx[m, len(curve[2][1:])]
                closed_gt_mask[m, :len(curve[2])] = 1


                '''
                # the rest of them!
                if len(curve[2]) > sn:
                    # take start/end points + sample (sn-2) points = sn points
                    try:
                        closed_gt_256_sn_idx[m, 1:sn] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:(sn-1)]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        down_sample_point_copy[closed_gt_256_sn_idx[m, 1:sn], :] = np.Inf
                    except:
                        print("NN for closed_gt_256_sn_idx was not successful. skip this.")
                        log_string("Type 17", log_fout)
                        log_string("NN for closed_gt_256_sn_idx was not successful. skip this.", log_fout)
                        closed_curve_NN_search_failed = True
                        break
                    #closed_gt_256_sn_idx[i, 63] = curve[2][-1]
                    closed_gt_mask[m, 0:sn] = 1
                else:
                    try:
                        closed_gt_256_sn_idx[m, 1:len(curve[2][1:])+1] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:len(curve[2][1:])]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        # make the point unavailable.
                        down_sample_point_copy[closed_gt_256_sn_idx[m, 1:len(curve[2][1:])+1], :] = np.Inf
                    except:
                        print("NN for closed_gt_256_sn_idx len() < sn was not successful. skip this.")
                        log_string("Type 18", log_fout)
                        log_string("NN for closed_gt_256_sn_idx len() < sn was not successful. skip this.", log_fout)
                        closed_curve_NN_search_failed = True
                        break
                    closed_gt_256_sn_idx[m, len(curve[2][1:])+1:] = closed_gt_256_sn_idx[m, len(curve[2][1:])]
                    closed_gt_mask[m, :len(curve[2])] = 1
                '''
                
                # closed_gt_type, closed_type_onehot
                if curve[0] == "Circle": closed_gt_type[m,0] = 1
                
                # closed_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[closed_gt_pair_idx[m,][0], ]
                closed_gt_res[m, ] = np.array([res1])

                # open_gt_sample_points
                closed_gt_sample_points[m, ...] = down_sample_point[closed_gt_256_sn_idx[m], ]
                m = m + 1
            if closed_curve_NN_search_failed: continue

            n = 0
            open_curve_NN_search_failed = False
            for curve in open_curves:
                down_sample_point_copy = down_sample_point.copy()
                if len(curve[2]) > sn:
                    # we can only take sn points for GT!
                    sample_num = sn
                    stacked_col = graipher_FPS_idx_collector(np.column_stack((np.array(curve[2]), vertices[curve[2], ...])), sample_num)
                    new_end = stacked_col[1, 0]
                    stacked_col[1, 0] = stacked_col[-1, 0]
                    stacked_col[-1, 0] = new_end
                    curve[2] = list(stacked_col[:, 0].astype(int))
                # first/last element
                try:
                    pair_idx = nearest_neighbor_finder(vertices[np.array([curve[2][0], curve[2][-1]]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    if (open_gt_pair_idx[0:n, :] == pair_idx).all(axis = 1).any() or (open_gt_pair_idx[0:n, :] == pair_idx[::-1]).all(axis = 1).any():
                        raise ValueError
                    open_gt_pair_idx[n,:] = pair_idx
                    open_gt_256_sn_idx[n, 0] = open_gt_pair_idx[n,0]
                    open_gt_valid_mask[n, 0] = 1
                    down_sample_point_copy[open_gt_pair_idx[n,0], :] = np.Inf
                    down_sample_point_copy[open_gt_pair_idx[n,1], :] = np.Inf
                except:
                    print("NN for open_gt_pair_idx was not successful. skip this.")
                    log_string("Type 29", log_fout)
                    log_string("NN for open_gt_pair_idx was not successful. skip this.", log_fout)
                    open_curve_NN_search_failed = True
                    break
                
                middle_idx_num = len(curve[2]) - 2
                #open_gt_256_sn_idx[n, :] = curve[2] + [curve[2][-1]]*(sn - indicies_num)
                try:
                    open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                    down_sample_point_copy[open_gt_256_sn_idx[n, 1:(middle_idx_num+1)],:] = np.Inf
                except:
                    print("NN for open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.")
                    log_string("Type 30", log_fout)
                    log_string("NN for open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.", log_fout)
                    open_curve_NN_search_failed = True
                    break
                open_gt_256_sn_idx[n, (middle_idx_num+1):sn] = open_gt_pair_idx[n, 1]
                open_gt_mask[n, 0:(middle_idx_num+2)] = 1

                '''
                # open_gt_256_sn_idx
                if len(curve[2]) > sn:
                    # sample start/end points + sample (sn-2) points = sn points
                    try:
                        open_gt_256_sn_idx[n, 1:(sn-1)] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))[:(sn-2)]),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        down_sample_point_copy[open_gt_256_sn_idx[n, 1:(sn-1)],:] = np.Inf
                    except:
                        print("NN for open_gt_256_sn_idx was not successful. skip this.")
                        log_string("Type 20", log_fout)
                        log_string("NN for open_gt_256_sn_idx was not successful. skip this.", log_fout)
                        open_curve_NN_search_failed = True
                        break
                    open_gt_256_sn_idx[n, (sn-1)] = open_gt_pair_idx[n,1]
                    open_gt_mask[n, 0:sn] = 1
                else:
                    middle_idx_num = len(curve[2]) - 2
                    #open_gt_256_sn_idx[n, :] = curve[2] + [curve[2][-1]]*(sn - indicies_num)
                    try:
                        open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))),:], down_sample_point_copy, use_clustering=False, neighbor_distance=1)
                        down_sample_point_copy[open_gt_256_sn_idx[n, 1:(middle_idx_num+1)],:] = np.Inf
                    except:
                        print("NN for open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.")
                        log_string("Type 21", log_fout)
                        log_string("NN for open_gt_256_sn_idx[n, 1:(middle_idx_num+1)] was not successful. skip this.", log_fout)
                        open_curve_NN_search_failed = True
                        break
                    open_gt_256_sn_idx[n, (middle_idx_num+1):sn] = open_gt_pair_idx[n, 1]
                    open_gt_mask[n, 0:(middle_idx_num+2)] = 1
                '''

                # open_gt_type, open_type_onehot BSpline, Lines, HalfCircle
                if curve[0] == 'BSpline': open_gt_type[n,0], open_type_onehot[n, ] = 1, np.array([0, 1, 0])
                elif curve[0] == 'Line' : open_gt_type[n,0], open_type_onehot[n, ] = 2, np.array([0, 0, 1]) # "Line"
                #1234elif curve[0] == 'HalfCircle' : open_gt_type[n,0], open_type_onehot[n, ] = 3, np.array([0, 0, 0, 1]) # "HalfCircle"
                
                # open_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[open_gt_pair_idx[n, ][0], ]
                res2 = vertices[curve[2][-1], ]-down_sample_point[open_gt_pair_idx[n, ][1], ]
                open_gt_res[n, ] = np.array([res1, res2]).flatten()

                # open_gt_sample_points
                open_gt_sample_points[n, ...] = down_sample_point[open_gt_256_sn_idx[n], ]
                n = n + 1
            if open_curve_NN_search_failed: continue


            # before saving this object
            # check..
            # 0. if the generated mesh looks fine
            # 1. if 256_sn open curves are correct
            # 1. if residual vectors are fine
            # 1. if pair index ok
            # 2. assert the labels
            '''
            if check_point3:

                # create updates
                def close_visualization(vis):
                    vis.close()

                s = 0
                # W
                def update_visualization_open_curve_forward(vis, vertices, down_sample_point, open_gt_256_sn_idx, open_gt_sample_points, open_curves):
                    global s

                    # asserts..
                    # open_gt_pair_idx = open_gt_256_sn_idx
                    assert open_gt_pair_idx[s, 0] == open_gt_256_sn_idx[s, 0]
                    assert open_gt_pair_idx[s, 1] == open_gt_256_sn_idx[s, -1]

                    # open_gt_valid_mask = open_gt_mask
                    
                    if open_gt_valid_mask[s, 0] == 0: 
                        assert open_gt_valid_mask[s, 0] == open_gt_mask[s, :].sum() == open_gt_pair_idx[s, :].sum() == open_gt_256_sn_idx[s, :].sum()

                    # open_gt_sample_points should be almost same.
                    assert np.mean(np.sqrt(np.sum((down_sample_point[open_gt_256_sn_idx[s, np.where(open_gt_mask[s, :] == 1)[0]], ...]- open_gt_sample_points[s, :][np.where(open_gt_mask[s, :] == 1)[0], ...])**2, axis = 1))) < 0.001

                    color_array = np.zeros_like(np.concatenate([down_sample_point, vertices]))
                    color_array[open_gt_256_sn_idx[s, :], :] = [0.0, 0.0, 0.99] # blue
                    color_array[open_gt_pair_idx[s, :], :] = [0.99, 0.0, 0.0] # red
                    color_array[FPS_num + np.array(open_curves[s][2]), :] = [0.0, 0.99, 0.0] # green

                    if open_gt_valid_mask[s+1, 0] == 1:
                        s += 1
                    point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, vertices]))
                    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()
                    #vis.run()

                # E
                def update_visualization_open_curve_backward(vis, vertices, down_sample_point, open_gt_256_sn_idx, open_gt_sample_points, open_curves):
                    global s
                    # asserts..
                    # open_gt_pair_idx = open_gt_256_sn_idx
                    assert s > 0
                    assert open_gt_pair_idx[s, 0] == open_gt_256_sn_idx[s, 0]
                    assert open_gt_pair_idx[s, 1] == open_gt_256_sn_idx[s, -1]

                    # open_gt_valid_mask = open_gt_mask
                    
                    if open_gt_valid_mask[s, 0] == 0: 
                        assert open_gt_valid_mask[s, 0] == open_gt_mask[s, :].sum() == open_gt_pair_idx[s, :].sum() == open_gt_256_sn_idx[s, :].sum()

                    # open_gt_sample_points should be almost same.
                    assert np.mean(np.sqrt(np.sum((down_sample_point[open_gt_256_sn_idx[s, np.where(open_gt_mask[s, :] == 1)[0]], ...]- open_gt_sample_points[s, :][np.where(open_gt_mask[s, :] == 1)[0], ...])**2, axis = 1))) < 0.001

                    color_array = np.zeros_like(np.concatenate([down_sample_point, vertices]))
                    color_array[open_gt_256_sn_idx[s, :], :] = [0.0, 0.0, 0.99] # blue
                    color_array[open_gt_pair_idx[s, :], :] = [0.99, 0.0, 0.0] # red
                    color_array[FPS_num + np.array(open_curves[s][2]), :] = [0.0, 0.99, 0.0] # green

                    if open_gt_valid_mask[s-1, 0] == 1:
                        s -= 1
                    point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, vertices]))
                    point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                    vis.update_geometry(point_cloud)
                    vis.poll_events()
                    vis.update_renderer()
                    #vis.run()


                # first visualization: down_sample_point, edge and corner points level and their residual vectors
                color_array = np.zeros_like(np.concatenate([down_sample_point, down_sample_point]))
                color_array[np.where(edge_points_label == 1)[0], :] = [0.0, 0.0, 0.99] # blue
                color_array[np.where(corner_points_label == 1)[0], :] = [0.99, 0.0, 0.0] # red
                color_array[FPS_num + np.where(edge_points_label == 1)[0], :] = [0.0, 0.99, 0.0]
                color_array[FPS_num + np.where(corner_points_label == 1)[0], :] = [0.0, 0.99, 0.0]

                #color_array[open_curves[s][2][1:-1], :] = [0.0, 0.99, 0.0]
                correct_edges = down_sample_point+edge_points_residual_vector
                correct_corners = down_sample_point+corner_points_residual_vector
                delta_edges = correct_edges - correct_corners

                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([down_sample_point, correct_corners + delta_edges]))
                point_cloud.colors = open3d.utility.Vector3dVector(color_array)

                vis = open3d.visualization.VisualizerWithKeyCallback()
                vis.create_window()
                vis.register_key_callback(87, partial(update_visualization_open_curve_forward, vertices = vertices, down_sample_point = down_sample_point, open_gt_256_sn_idx = open_gt_256_sn_idx, open_gt_sample_points = open_gt_sample_points, open_curves = open_curves)) # W    
                vis.register_key_callback(69, partial(update_visualization_open_curve_backward, vertices = vertices, down_sample_point = down_sample_point, open_gt_256_sn_idx = open_gt_256_sn_idx, open_gt_sample_points = open_gt_sample_points, open_curves = open_curves)) # E
                vis.register_key_callback(81, close_visualization) # Q
                vis.add_geometry(point_cloud)
                vis.run()
            '''

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
                  ('open_gt_256_sn_idx', 'O'),
                ('closed_gt_256_sn_idx', 'O'),
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

        #list_obj_line = list_obj_file.readline()
        #list_ftr_line = list_ftr_file.readline()

    if batch_count > 0:
        file_ = save_prefix+"_"+str(file_count)+"_end"+".mat"
        data['batch_count'] = batch_count
        scipy.io.savemat(file_, data)
    

    list_obj_file.close()
    list_ftr_file.close()
    log_fout.close()



