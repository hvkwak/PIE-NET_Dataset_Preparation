import os
import sys
import numpy as np
import trimesh
from utils import delete_newline
from utils import delete_spaces
from utils import sharp_edges
from utils import curves_with_vertex_indices
from utils import calc_distances
from utils import log_string
from utils import nearest_neighbor_finder
from utils import graipher_FPS
from utils import label_creator
from utils import residual_vector_creator
from utils import view_point
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
        has_sharp_edge = sharp_edges(list_ftr_line)
        
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
                
                # Curves with vertex indices: sharp edges of BSpline, Line, Cycle only.
                sharp_curves = curves_with_vertex_indices(list_ftr_line)

                # (Optional) Filter out/Classify accordingly the curves such as:
                # 1. Filter out Circles with the different endpoints.
                # 2. Classify two BSplines that make a circle(same endpoints) as closed curve.
                # 3. Filter out Several BSplines can make a closed curve.
                # Note: We just implement the first option. Keep options above in mind for later use.
                for curve in sharp_curves:
                    if curve[0] == 'Circle' and curve[1][0] != curve[1][-1]:
                        if calc_distances(vertices[curve[1][0], :], vertices[curve[1][-1], :] ) > 1.0:
                            # Not even a slight (hand)labeling error. We remove this curve(circle.
                            log_string('Curves in the circle do not match. Skip this curve: '+str(curve), log_fout)
                            sharp_curves.remove(curve)


                # Curve Classification
                # Open: BSpline, Line
                # Closed: Circle
                #
                # Edge/Corner points Classification
                # Corner Points: Start and end points of open curve, Start and end points of Lines
                # Edge Points: All the vertices of open or closed curve, All the vertices of a line
                open_curves = []
                closed_curves = []
                corner_points_ori = []
                edge_points_ori = []
                for curve in sharp_curves:
                    if curve[0] == 'BSpline': # Open
                        open_curves.append(curve)
                        corner_points_ori.append(curve[1][0])
                        corner_points_ori.append(curve[1][-1])
                        edge_points_ori =  edge_points_ori + curve[1][:]
                    elif curve[0] == 'Circle': # Closed
                        closed_curves.append(curve)
                        edge_points_ori =  edge_points_ori + curve[1][:]
                    elif curve[0] == 'Line': # Open
                        open_curves.append(curve)
                        corner_points_ori.append(curve[1][0])
                        corner_points_ori.append(curve[1][-1])
                        edge_points_ori =  edge_points_ori + curve[1][:]

                # Downsampling
                # create mesh
                mesh = trimesh.Trimesh(vertices = vertices, faces = faces, vertex_normals = vertex_normals)

                # (uniform) random sample 100K surface points: Points in space on the surface of mesh
                mesh_sample_xyz, mesh_sample_idx = trimesh.sample.sample_surface(mesh, 100000)

                # (greedy) Farthest Points Sampling
                down_sample_point = graipher_FPS(mesh_sample_xyz, FPS_num) # dtype: np.float64
                view_point(down_sample_point)
                # Annotation transfer
                # edge_points_now ('PC_8096_edge_points_label_bin'), (8096, 1), dtype: uint8
                # Note: find a nearest neighbor of each edge_point in edge_points_ori, label it as an "edge"
                nearest_neighbor_idx_edge = nearest_neighbor_finder(vertices[edge_points_ori,:], down_sample_point)
                nearest_neighbor_idx_corner = nearest_neighbor_finder(vertices[corner_points_ori,:], down_sample_point)

                edge_points_label_bin = label_creator(FPS_num, nearest_neighbor_idx_edge)
                edge_points_res_vec = residual_vector_creator(down_sample_point, edge_points_label_bin, vertices, edge_points_ori)

                corner_points_label_bin = label_creator(FPS_num, nearest_neighbor_idx_corner)
                

        list_obj_line = list_obj_file.readline()
        list_ftr_line = list_ftr_file.readline()

    list_obj_file.close()
    list_ftr_file.close()
    log_fout.close()


if __name__ == "__main__": 
    main()
    