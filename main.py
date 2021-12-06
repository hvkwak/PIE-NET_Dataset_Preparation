import os
import sys
import numpy as np
from numpy.core.fromnumeric import mean
import trimesh
import scipy.io
import random
from tqdm import tqdm
from utils import delete_newline
from utils import curves_with_vertex_indices
from utils import cross_points_finder
from utils import update_lists_open
from utils import another_half_curve_pair_exist
from utils import graipher_FPS
from utils import nearest_neighbor_finder
#from utils import greedy_nearest_neighbor_finder
from utils import log_string
from utils import merge_two_half_circles_or_BSpline
from utils import update_lists_closed
from utils import half_curves_finder
from utils import touch_in_circles_or_BSplines
from utils import degrees_same
from cycle_detection import cycle_detection_in_BSplines
from grafkom1Framework import ObjLoader
from visualizer import view_point


def main():
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
    #FPS_num = int(args[3]) 8096
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
    data = {'Training_data': np.zeros((64, 1), dtype = object)}
    curve_num_list = []
    for i in range(244, model_total_num):
        
        # check the feature file if it contains at least a sharp edges
        # and check that models are same.
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

            # load the object file: all vertices / faces of a Model with at least one sharp edge.
            Loader = ObjLoader(list_obj_line)
            vertices = np.array(Loader.vertices)
            faces = np.array(Loader.faces)
            vertex_normals = np.array(Loader.vertex_normals)
            del Loader

            if vertices.shape[0] > 30000: # make sure we have < 30K vertices to keep it simple.
                print("vertices > 30000. skip this.")
                log_string("vertices > 30000. skip this.", log_fout)
                del vertices
                del faces
                del vertex_normals
                continue
            
            # Curves with vertex indices: (sharp and not sharp)edges of BSpline, Line, Cycle only.
            all_curves = []
            try:
                all_curves = curves_with_vertex_indices(list_ftr_line)
            except:
                print("there are curves not in [Circle, BSpline, Line]. skip this.")
                log_string("there are curves not in [Circle, BSpline, Line]. skip this.", log_fout)
                continue                        
            curve_num = len(all_curves)

            # Preprocessing: (Re)classify / Filter points accordingly!
            open_curves = []
            closed_curves = []
            corner_points_ori = []
            edge_points_ori = []
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
            
            # Find a misclassified BSplines and first classify them correctly into circle,
            # if they have same start and end points.
            i = 0
            BSpline_list_num = len(BSpline_list)
            while i < BSpline_list_num :
                if BSpline_list[i][2][0] == BSpline_list[i][2][-1]:
                    BSpline_list[i][0] = 'Circle'
                    Circle_list.append(BSpline_list[i])
                    del BSpline_list[i]
                    BSpline_list_num = BSpline_list_num - 1
                    i = i - 1
                i = i + 1
            
            # Check if there are half Circles/BSplines pair, merge them if there's one.
            #BSpline_list.append(['BSpline', 3, [33, 99, 66, 55, 44, 11, 22, 77]] )
            #BSpline_list.append(['BSpline', 3, [77, 99, 66, 55, 44, 11, 22, 33]] )
            BSpline_list = half_curves_finder(BSpline_list)
            Circle_list = half_curves_finder(Circle_list)

            # Move Circles in BSplines to Circles.
            i = 0
            BSpline_list_num = len(BSpline_list)
            while i < BSpline_list_num:
                if BSpline_list[i][0] == 'Circle':
                    Circle_list.append(BSpline_list[i])
                    del BSpline_list[i]
                    BSpline_list_num = BSpline_list_num - 1
                    i = i - 1
                i = i + 1
            
            # There are still open curves in Circles. keep them as BSpline.
            i = 0
            Circle_list_num = len(Circle_list)
            while i < Circle_list_num:
                if Circle_list[i][2][0] != Circle_list[i][2][-1]:
                    Circle_list[i][0] = 'BSpline'
                    BSpline_list.append(Circle_list[i])
                    del Circle_list[i]
                    Circle_list_num = Circle_list_num - 1
                    i = i - 1
                i = i + 1
            
            # Find BSplines with degree = 1 and classify them accordingly:
            # BSpline with degree = 1 and both start/end points touch circles, remove it from the list
            # BSpline with degree = 1 and no touches -> keep them as line.

            # first take all the BSplines of degree 1
            BSpline_degree_one_list = []
            BSpline_list_num = len(BSpline_list)
            i = 0
            while i < BSpline_list_num:
                if BSpline_list[i][1] == 1:
                    BSpline_degree_one_list.append(BSpline_list[i])
                    del BSpline_list[i]
                    i = i - 1
                    BSpline_list_num = BSpline_list_num - 1
                i = i + 1

            # delete them or add them to Lines.
            # touching two Circles(or BSplines) should be eliminated.
            BSpline_degree_one_list_num = len(BSpline_degree_one_list)
            i = 0
            while i < BSpline_degree_one_list_num:
                if BSpline_degree_one_list[i][1] == 1 and touch_in_circles_or_BSplines(BSpline_degree_one_list[i][2], Circle_list+BSpline_list):
                    del BSpline_degree_one_list[i]
                    i = i - 1
                    BSpline_degree_one_list_num = BSpline_degree_one_list_num - 1
                elif BSpline_degree_one_list[i][1] == 1 and not touch_in_circles_or_BSplines(BSpline_degree_one_list[i][2], Circle_list+BSpline_list):
                    BSpline_degree_one_list[i][0] = 'Line'
                    Line_list.append(BSpline_degree_one_list[i])
                    del BSpline_degree_one_list[i]
                    i = i - 1
                    BSpline_degree_one_list_num = BSpline_degree_one_list_num - 1
                i = i + 1

            Line_list_num = len(Line_list)
            i = 0
            while i < Line_list_num:
                if touch_in_circles_or_BSplines(Line_list[i][2], Circle_list+BSpline_list):
                    del Line_list[i]
                    i = i - 1
                    Line_list_num = Line_list_num - 1
                i = i + 1
            
            # Check if multiple BSplines can form a circle.
            # first check if there are vertices that are "visited" more than twice. 
            visited_verticies = []
            BSpline_list_num = len(BSpline_list)
            
            for i in range(BSpline_list_num):
                visited_verticies.append(BSpline_list[i][2][0])
                visited_verticies.append(BSpline_list[i][2][-1])
            
            if (np.bincount(visited_verticies) > 2).sum() > 0:
                print("there exist at least one vertex that is visited more than twice. skip this.")
                log_string("there exist at least one vertex that is visited more than twice. skip this.", log_fout)
                continue
            
            # if there are at least one detected cycle, skip it.
            detected_cycles_in_BSplines = cycle_detection_in_BSplines(BSpline_list)
            if len(detected_cycles_in_BSplines) > 0 :
                print("There are at least one detected cycle, skip this.")
                log_string("there are at least one detected cycle, skip this.", log_fout)
                continue                
                
            '''
            i = 0
            while i < detected_cycles_in_BSplines_num: # e.g.) [[3, 4, 5, 6], [11, 12, 13]]
                parts_num = len(detected_cycles_in_BSplines[i]) # e.g) Cycle Number 1: 3, 4, 5, 6
                temp_Splines = []
                elements_num = 0

                # for cycle number i:
                k = 0
                BSpline_list_num = len(BSpline_list)
                while k < BSpline_list_num:
                    if BSpline_list[k][2][0] in detected_cycles_in_BSplines[i] and \
                        BSpline_list[k][2][-1] in detected_cycles_in_BSplines[i]:
                        elements_num = elements_num + len(BSpline_list[k][2])
                        temp_Splines.append(BSpline_list[k])
                        del BSpline_list[k]
                        k = k - 1
                        BSpline_list_num = BSpline_list_num - 1
                    k = k + 1
                


                if not degrees_same(temp_Splines):
                    print("BSplines of this detected cycle have different degrees. skip these Splines")
                    continue
                
                avg_length = elements_num / np.float32(parts_num)
                different_length = False
                # check if the splines have similar length. not too similar -> continue.
                for n in range(parts_num):
                    if np.abs(len(temp_Splines[n]) - avg_length) > 2:
                        different_length = True
                if different_length:
                    print("Splines for this cycle have too different length. skip this.")
                    log_string("Splines for this cycle have too different length. skip this.", log_fout)
                
            
                centroid_x = 0.0
                centroid_y = 0.0
                centroid_z = 0.0
                vertices_num = 0
                # for all Splines of a cycle:
                for k in range(parts_num):
                    vertices_num += len(temp_Splines[k])
                    x, y, z = vertices[temp_Splines[k], :].sum(axis = 1)
                    centroid_x += x
                    centroid_y += y
                    centroid_z += z
                
                centroid_x = centroid_x / np.float32(vertices_num)
                centroid_y = centroid_y / np.float32(vertices_num)
                centroid_z = centroid_z / np.float32(vertices_num)
                
                distances = np.sum(((centroid_x, centroid_y, centroid_z) - vertices[temp_Splines[k], :])**2, axis = 0)
                mean_distance = np.mean(distances)
                
                merged_vertices_list = []
                if ((mean_distance - distances)**2 < 1).sum() == distances.shape[0]:
                    print("A possible circle found.")
                    start_vertice_num_of_circle = detected_cycles_in_BSplines[i][0]
                    next_start_vertice_num_of_circle = None
                    while len(temp_Splines) != 0:
                        for m in range(len(temp_Splines)): 
                            if temp_Splines[m][2][0] == start_vertice_num_of_circle:
                                merged_vertices_list += temp_Splines[m][2]
                                next_vertice_num_of_circle = temp_Splines[m][2][-1]
                            if temp_Splines[m][2][0] == next_vertice_num_of_circle:
                                merged_vertices_list += temp_Splines[m][2][1:]
                                next_vertice_num_of_circle = temp_Splines[m][2][-1]
                Circle_list.append(['Circle', None, merged_vertices_list])
            '''


            print("Visualizing.. all_curves_list")
            view_point(vertices, all_curves)
            
            if len(Circle_list) > 0:
                print("Visualizing.. Circle_list")
                view_point(vertices, Circle_list)            
            
            if len(BSpline_list) > 0:
                print("Visualizing.. BSpline_list")
                view_point(vertices, BSpline_list)
            
            if len(Line_list) > 0:
                print("Visualizing.. Lines_list")
                view_point(vertices, Line_list)
            
            continue
            
            curve_num_list.append(curve_num)
            curve_num = len(all_curves)
            k = 0
            while k < curve_num:
                # Note that this is a mutable object which is in list.
                curve = all_curves[k]
                circle_pair_index = [None]
                circle_pair_Forward = [None]

                # check if there are (corner) points, where two curves cross or meet.
                if len(curve[2]) > 2 and k < curve_num-1:
                    for j in range(k+1, curve_num):
                        if len(all_curves[j][2]) > 2:
                            cross_points = cross_points_finder(curve[2], all_curves[j][2])
                            if len(cross_points) > 0:
                                log_string("len(cross_points) > 0.", log_fout)
                                corner_points_ori = corner_points_ori + cross_points

                # classifications
                if curve[0] == 'Line':
                    open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                elif curve[0] == 'Circle':
                    if curve[2][0] != curve[2][-1] and another_half_curve_pair_exist(curve, all_curves[k+1:], circle_pair_index, circle_pair_Forward):
                        # this one consist of a pair of two half-circle curves!
                        curve = merge_two_half_circles_or_BSpline(curve, k+1+circle_pair_index[0], all_curves, circle_pair_Forward)
                        closed_curves, edge_points_ori = update_lists_closed(curve, closed_curves, edge_points_ori)
                        del all_curves[k+1+circle_pair_index[0]]
                        circle_pair_index = [None]
                        circle_pair_Forward = [None]
                        curve_num = curve_num - 1
                    else:
                        closed_curves, edge_points_ori = update_lists_closed(curve, closed_curves, edge_points_ori)
                elif curve[0] == 'BSpline':
                    if curve[2][0] != curve[2][-1] and another_half_curve_pair_exist(curve, all_curves[k+1:], circle_pair_index, circle_pair_Forward):
                        curve = merge_two_half_circles_or_BSpline(curve, k+1+circle_pair_index[0], all_curves, circle_pair_Forward)
                        closed_curves, edge_points_ori = update_lists_closed(curve, closed_curves, edge_points_ori)
                        del all_curves[k+1+circle_pair_index[0]]
                        circle_pair_index = [None]
                        circle_pair_Forward = [None]
                        curve_num = curve_num - 1
                    else:
                        open_curves, corner_points_ori, edge_points_ori = update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori)
                '''
                
                if curve[0] == 'BSpline' or curve[0] == 'Line':
                    open_curves, corner_points_ori, edge_points_ori = update_lists(curve, open_curves, corner_points_ori, edge_points_ori)
                elif curve[0] == 'Circle': # Closed
                    if curve[2][0] != curve[2][-1] and another_half_curve_pair_exist(curve, all_curves[k:], circle_pair_index):
                        # this one consist of a pair of two half-circle curves!
                        all_curves[k+circle_pair_index[0]][0] = 'BSpline' # change the other to BSpline.
                        curve[0] = 'BSpline' # change it to BSpline.
                        
                    else:
                        closed_curves.append(curve)
                        edge_points_ori =  edge_points_ori + curve[2][:]
                '''
                k = k + 1
            del all_curves


            # if there are more than 256 curves in each section: don't use this model.
            if (len(open_curves) > 256) or (len(closed_curves) > 256): 
                print("(open/closed)_curves > 256. skip this.")
                log_string("(open/closed)_curves > 256. skip this.", log_fout)
                continue

            if (len(open_curves) == 0) or (len(closed_curves) == 0): 
                print("(open/closed)_curves = 0. skip this.")
                log_string("(open/closed)_curves = 0. skip this.", log_fout)
                continue

            # make the list unique
            edge_points_ori = np.unique(edge_points_ori)
            corner_points_ori = np.unique(corner_points_ori)
            skip_this_model = edge_points_ori.shape[0] == 0 or corner_points_ori.shape[0] == 0 \
                            or edge_points_ori.shape[0] > FPS_num or  edge_points_ori.shape[0] > FPS_num

            if skip_this_model: 
                print("problems in (edge/corner)_points_ori. Skip this.")
                log_string("problems in (edge/corner)_points_ori. Skip this.", log_fout)
                continue

            # Downsampling
            # create mesh
            mesh = trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = vertex_normals)

            # (uniform) random sample 100K surface points: Points in space on the surface of mesh
            mesh_sample_xyz, _ = trimesh.sample.sample_surface(mesh, 100000)
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
                nearest_neighbor_idx_edge_1 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=False, neighbor_distance=0.5)
                nearest_neighbor_idx_corner_1 = nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point, use_clustering=False, neighbor_distance=0.5)
                distance_max_1 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_1, :])**2).sum(axis = 1))
                if distance_max_1 > 1: 
                    print("distance_max_1 > 1. skip this.")
                    log_string("distance_max_1 > 1. skip this.", log_fout)
                    continue
                nearest_neighbor_idx_edge = nearest_neighbor_idx_edge_1
                nearest_neighbor_idx_corner = nearest_neighbor_idx_corner_1
            except:
                print("NN was not successful. skip this.")
                log_string("NN was not successful. skip this.", log_fout)
                continue

            # option 2 : clustering of bins
            # First build a cluster nearby multiple ties.
            '''
            try:
                best_max = np.Inf
                for l in np.arange(0.8, 1.2, 0.1):
                    neighbor_distance = l
                    nearest_neighbor_idx_edge_2 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=True, neighbor_distance=neighbor_distance)
                    distance_max_2 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_2, :])**2).sum(axis = 1))
                    if distance_max_2 < best_max:
                        best_neighbor_distance = l

                neighbor_distance = best_neighbor_distance
                nearest_neighbor_idx_edge_2 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=True, neighbor_distance=neighbor_distance)
                nearest_neighbor_idx_corner_2 = nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point, use_clustering=True, neighbor_distance=neighbor_distance)
                distance_max_2 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_2, :])**2).sum(axis = 1))    
                log_string('distance_max_2: '+str(distance_max_2), log_fout)
                if distance_max_2 > 10: 
                    print("distance_max_2 > 10. skip this.")
                    continue
            except:
                print("nearest_neighbor_finder was not successful. skip this.")
                continue
            nearest_neighbor_idx_edge = nearest_neighbor_idx_edge_2
            nearest_neighbor_idx_corner = nearest_neighbor_idx_corner_2
            '''

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
            open_gt_type = np.zeros((256, 1), dtype=np.uint8) # Note: BSpline and Lines, so two label types: 1, 2
            open_type_onehot = np.zeros((256, 4), dtype=np.uint8)
            open_gt_res = np.zeros((256, 6), dtype=np.float32)
            open_gt_sample_points = np.zeros((256, 64, 3), dtype=np.float32)
            open_gt_mask = np.zeros((256, 64), dtype=np.uint8)
            closed_gt_256_64_idx = np.zeros((256, 64), dtype=np.uint8)
            closed_gt_mask = np.zeros((256, 64), dtype=np.uint8)
            closed_gt_type = np.zeros((256, 1), dtype=np.uint8)
            closed_gt_res = np.zeros((256, 3), dtype=np.float32)
            closed_gt_sample_points = np.zeros((256, 64, 3), dtype=np.uint8)
            closed_gt_valid_mask = np.zeros((256, 1), dtype=np.uint8)
            closed_gt_pair_idx = np.zeros((256, 1), dtype=np.uint16)
            
            # and compute them
            # down_sample_point is already there.
            edge_points_label[nearest_neighbor_idx_edge] = 1
            edge_points_residual_vector[nearest_neighbor_idx_edge, :] = vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge, :]
            corner_points_label[nearest_neighbor_idx_corner, ] = 1
            corner_points_residual_vector[nearest_neighbor_idx_corner, ] = vertices[corner_points_ori,:] - down_sample_point[nearest_neighbor_idx_corner, :]
            
            m = 0
            for curve in closed_curves:
                try:
                    closed_gt_pair_idx[m,0] = nearest_neighbor_finder(vertices[np.array([curve[2][0]]),:], down_sample_point, use_clustering=False, neighbor_distance=1)
                except:
                    print("NN for closed_gt_pair_idx was not successful. skip this.")
                    log_string("NN for closed_gt_pair_idx was not successful. skip this.", log_fout)
                    continue
                closed_gt_valid_mask[m, 0] = 1
                # closed_gt_256_64_idx
                if curve[2][0] == curve[2][-1]: curve[2] = curve[2][:-1] # update if these two indicies are same.
                if len(curve[2]) > 64:
                    # take start/end points + sample 62 points = 64 points
                    closed_gt_256_64_idx[m, 0] = curve[2][0]
                    try:
                        closed_gt_256_64_idx[m, 1:64] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:], len(curve[2][1:]))[:63]),:], down_sample_point, use_clustering=False, neighbor_distance=1)
                    except:
                        print("NN for closed_gt_256_64_idx was not successful. skip this.")
                        log_string("NN for closed_gt_256_64_idx was not successful. skip this.", log_fout)
                        continue                        
                    #closed_gt_256_64_idx[i, 63] = curve[2][-1]
                    closed_gt_mask[m, 0:64] = 1
                else:
                    indicies_num = len(curve[2])
                    closed_gt_mask[m, 0:indicies_num] = 1
                    closed_gt_256_64_idx[m, :] = curve[2] + [curve[2][-1]]*(64 - indicies_num)

                # closed_gt_type, closed_type_onehot
                if curve[0] == "Circle": closed_gt_type[m,0] = 1
                
                # open_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[closed_gt_pair_idx[m, ][0], ]
                closed_gt_res[m, ] = np.array([res1])

                # open_gt_sample_points
                closed_gt_sample_points[m, ...] = down_sample_point[closed_gt_256_64_idx[m], ]
                m = m + 1

            n = 0
            for curve in open_curves:
                try:
                    open_gt_pair_idx[n, ] = nearest_neighbor_finder(vertices[np.array([curve[2][0], curve[2][-1]]),:], down_sample_point, use_clustering=False, neighbor_distance=1)
                except:
                    print("NN for open_gt_pair_idx was not successful. skip this.")
                    log_string("NN for open_gt_pair_idx was not successful. skip this.", log_fout)
                    continue
                open_gt_valid_mask[n, 0] = 1
                # open_gt_256_64_idx
                if len(curve[2]) > 64:
                    # take start/end points + sample 62 points = 64 points
                    open_gt_256_64_idx[n, 0] = curve[2][0]
                    try:
                        open_gt_256_64_idx[n, 1:63] = nearest_neighbor_finder(vertices[np.array(random.sample(curve[2][1:-1], len(curve[2][1:-1]))[:62]),:], down_sample_point, use_clustering=False, neighbor_distance=1)
                    except:
                        print("NN for open_gt_256_64_idx was not successful. skip this.")
                        log_string("NN for open_gt_256_64_idx was not successful. skip this.", log_fout)
                        continue
                    open_gt_256_64_idx[n, 63] = curve[2][-1]
                    open_gt_mask[n, 0:64] = 1
                else:
                    indicies_num = len(curve[2])
                    open_gt_mask[n, 0:indicies_num] = 1
                    open_gt_256_64_idx[n, :] = curve[2] + [curve[2][-1]]*(64 - indicies_num)

                # open_gt_type, open_type_onehot
                if curve[0] == "BSpline": open_gt_type[n,0], open_type_onehot[n, ] = 1, np.array([0, 1, 0, 0])
                else: open_gt_type[n,0], open_type_onehot[n, ] = 2, np.array([0, 0, 1, 0]) # "Line"
                
                # open_gt_res
                res1 = vertices[curve[2][0], ]-down_sample_point[open_gt_pair_idx[n, ][0], ]
                res2 = vertices[curve[2][-1], ]-down_sample_point[open_gt_pair_idx[n, ][1], ]
                open_gt_res[n, ] = np.array([res1, res2]).flatten()

                # open_gt_sample_points
                open_gt_sample_points[n, ...] = down_sample_point[open_gt_256_64_idx[n], ]
                n = n + 1
            
            print("Ok. save data.")
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
            scipy.io.savemat(file_, data)
            print(file_, "saved.")
            log_string(str(file_) + " saved.", log_fout)
            batch_count = 0
            file_count = file_count + 1
            data = {'Training_data': np.zeros((64, 1), dtype = object)}
        else:
            batch_count = batch_count + 1

        list_obj_line = list_obj_file.readline()
        list_ftr_line = list_ftr_file.readline()

    list_obj_file.close()
    list_ftr_file.close()
    log_fout.close()


if __name__ == "__main__": 
    main()
    
