import os
import sys
import numpy as np
import trimesh
from tqdm import tqdm
from utils import delete_newline
from utils import curves_with_vertex_indices
from utils import cross_points_finder
from utils import update_lists
from utils import another_half_curve_pair_exist
from utils import graipher_FPS
from utils import nearest_neighbor_finder
from utils import greedy_nearest_neighbor_finder


from grafkom1Framework import ObjLoader


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
    list_obj_file = open(args[0], "r")
    list_ftr_file = open(args[1], "r")
    list_stt_file = open(args[2], "r")
    FPS_num = int(args[3])
    log_dir = args[4]
    log_fout = open(os.path.join(log_dir, 'generate_dataset_log.txt'), 'w')

    # readlines(of files) to make sure they are same length
    list_obj_lines = list_obj_file.readlines()
    list_ftr_lines = list_ftr_file.readlines()
    list_stt_lines = list_stt_file.readlines()
    model_total = len(list_obj_lines)
    
    for i in range(len(list_ftr_lines)):

        # check the feature file if it contains at least a sharp edge
        # and check that models are same.
        model_name_obj = "_".join(list_obj_lines[i].split('_')[0:2])
        model_name_ftr = "_".join(list_ftr_lines[i].split('_')[0:2])
        list_obj_line = delete_newline(list_obj_lines[i])
        list_ftr_line = delete_newline(list_ftr_lines[i])
        
        #has_sharp_edge = sharp_edges(list_ftr_line)
        has_sharp_edge = True
        
        if has_sharp_edge and model_name_obj == model_name_ftr:
            # make sure that there's no "\n" in the line.
            print("Processing: ", "_".join(list_ftr_lines[i].split('_')[0:2]), \
                ".........."+str(i+1) + "/" + str(model_total), "\n")
            use_this_model = True
            model_name = model_name_obj

            # load the object file: all vertices / faces of a Model with at least one sharp edge.
            Loader = ObjLoader(list_obj_line)
            vertices = np.array(Loader.vertices)
            faces = np.array(Loader.faces)
            vertex_normals = np.array(Loader.vertex_normals)

            if vertices.shape[0] < 30000: # make sure we have < 30K vertices to keep it simple.
                
                # Curves with vertex indices: (sharp)edges of BSpline, Line, Cycle only.
                all_curves = curves_with_vertex_indices(list_ftr_line)

                # (Optional) Filter out/Classify accordingly the curves such as:
                # 1. Filter out Circles with the different endpoints.
                # 2. Classify two BSplines that make a circle(same endpoints) as closed curve.
                # 3. Filter out Several BSplines can make a closed curve.
                # Note: We just implement the first option. Keep options above in mind for later use.
                '''
                for curve in sharp_curves:
                    if curve[0] == 'Circle' and curve[1][0] != curve[1][-1]:
                        if calc_distances(vertices[curve[1][0], :], vertices[curve[1][-1], :] ) > 1.0:
                            # Not even a slight (hand)labeling error. We remove this curve(circle.
                            log_string('Curves in the circle do not match. Skip this curve: '+str(curve), log_fout)
                            sharp_curves.remove(curve)
                '''


                # Classifications
                # Open Curves: BSplines and Lines
                # Closed Curves: Circles
                # Edge Points: All the vertices of open or closed curve, All the vertices of a line
                # Corner Points: Start and end points of open curve, Start and end points of Lines
                # Note: some circles are just divded into two circles with matching endpoints.
                # These circles should be one circle and added to closed_curves.
                # Since we'd be dealing with other datasets, accordingly, we keep them as BSpline.
                open_curves = []
                closed_curves = []
                corner_points_ori = []
                edge_points_ori = []
                curve_num = len(all_curves)

                for k in range(curve_num):
                    
                    # Note that this is a mutable object which is in list.
                    curve = all_curves[k]
                    circle_pair_index = [None]

                    # check if there are (corner) points, where two curves cross or meet.
                    if len(curve[1]) > 2 and k < curve_num-1:
                        for j in range(k+1, curve_num):
                            if len(all_curves[j][1]) > 2:
                                cross_points = cross_points_finder(curve[1], all_curves[j][1])
                                corner_points_ori = corner_points_ori + cross_points

                    # classifications
                    if curve[0] == 'BSpline' or curve[0] == 'Line':
                        open_curves, corner_points_ori, edge_points_ori = update_lists(curve, open_curves, corner_points_ori, edge_points_ori)
                    elif curve[0] == 'Circle': # Closed
                        if curve[1][0] != curve[1][-1] and another_half_curve_pair_exist(curve, all_curves[k:], circle_pair_index):
                            # this one consist of a pair of two half-circle curves!
                            all_curves[k+circle_pair_index[0]][0] = 'BSpline' # change the other to BSpline.
                            curve[0] = 'BSpline' # change it to BSpline.
                            open_curves, corner_points_ori, edge_points_ori = update_lists(curve, open_curves, corner_points_ori, edge_points_ori)
                        else:
                            closed_curves.append(curve)
                            edge_points_ori =  edge_points_ori + curve[1][:]
                    k = k + 1

                #view_point(vertices[edge_points_ori,:])

                # make the list unique
                edge_points_ori = np.unique(edge_points_ori)
                corner_points_ori = np.unique(corner_points_ori)

                # Downsampling
                # create mesh
                mesh = trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = vertex_normals)

                # (uniform) random sample 100K surface points: Points in space on the surface of mesh
                mesh_sample_xyz, mesh_sample_idx = trimesh.sample.sample_surface(mesh, 100000)

                # (greedy) Farthest Points Sampling
                down_sample_point = graipher_FPS(mesh_sample_xyz, FPS_num) # dtype: np.float64
                
                
                # Annotation transfer
                # edge_points_now ('PC_8096_edge_points_label_bin'), (8096, 1), dtype: uint8
                # Note: find a nearest neighbor of each edge_point in edge_points_ori, label it as an "edge"
                '''
                # option 1 : no clustering, just take nearest neighbors. Ties shoud be handled again with nearest neighbor
                # concept around the tie point of a down_sample_point
                nearest_neighbor_idx_edge_1 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=False, neighbor_distance=1)
                distance_mean_1 = np.mean(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_1, :])**2).sum(axis = 1))
                distance_max_1 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_1, :])**2).sum(axis = 1))

                # option 2 : clustering of bins
                # grid search of neighbor_distance can make this slightly better than just keeping it as a hyperparameter.
                # Near "multiple" ties builds a cluster, and builds a neighborhood. 
                best_avg_max = np.Inf
                best_neighbor_distance = np.Inf
                for i in np.arange(0.8, 1.2, 0.05):
                    neighbor_distance = i
                    nearest_neighbor_idx_edge_2 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=True, neighbor_distance=neighbor_distance)
                    distance_max_2 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_2, :])**2).sum(axis = 1))
                    if distance_max_2 < best_avg_max:
                        best_neighbor_distance = i

                neighbor_distance = best_neighbor_distance
                nearest_neighbor_idx_edge_2 = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point, use_clustering=True, neighbor_distance=neighbor_distance)
                distance_mean_2 = np.mean(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_2, :])**2).sum(axis = 1))
                distance_max_2 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_2, :])**2).sum(axis = 1))    
                '''
                
                # option 3: greedy. Just random shuffle the indicies and take distance matrix and take minimums.
                nearest_neighbor_idx_edge_3 = greedy_nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point)                
                distance_mean_3 = np.mean(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_3, :])**2).sum(axis = 1))
                distance_max_3 = np.max(((vertices[edge_points_ori,:] - down_sample_point[nearest_neighbor_idx_edge_3, :])**2).sum(axis = 1))

                #print("distance_mean_1: ", distance_mean_1)
                #print("distance_max_1: ", distance_max_1)                
                #print("distance_mean_2 with neighbor_distance of ", neighbor_distance, ":", distance_mean_2)
                #print("distance_max_2 with neighbor_distance of ", neighbor_distance, ":", distance_max_2)                
                print("distance_mean_3: ", distance_mean_3)
                print("distance_max_3: ", distance_max_3)
                
                
                
                
                
                
                #nearest_neighbor_idx_corner = greedy_nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point)

                #edge_points_label_bin = label_creator(FPS_num, nearest_neighbor_idx_edge)
                #edge_points_res_vec = residual_vector_creator(down_sample_point, edge_points_label_bin, vertices, edge_points_ori)

                #corner_points_label_bin = label_creator(FPS_num, nearest_neighbor_idx_corner)
                

        list_obj_line = list_obj_file.readline()
        list_ftr_line = list_ftr_file.readline()

    list_obj_file.close()
    list_ftr_file.close()
    log_fout.close()


if __name__ == "__main__": 
    main()
    