import ast
import time
import numpy as np
from tqdm import tqdm
import open3d

'''
Prereq.
1. trimesh
2. tqdm
3. ast
'''
def view_point(points):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    point_cloud.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([point_cloud])

def timeit(func):
    """A wrapper function which calculates time for each execution of a function. 
    Code by Graipher from https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        func ([function]): A function being executed
    """
    def wrapper(*args, **kwargs):
        starting_time = time.perf_counter() # time.clock() since Python3.3 deprecated
        result = func(*args, **kwargs)
        ending_time = time.perf_counter()
        print('Duration: {}'.format(ending_time - starting_time))
        return result
    return wrapper

def log_string(out_str, log_fout):
    log_fout.write(out_str+'\n')
    log_fout.flush()

def residual_vector_creator(down_sample_point, edge_points_label_bin, vertices, edge_points_ori):
    """returns the residual vectors between edge points: down_sample_point[edge_points_label_bin] and vertices[edge_points_ori, :]

    Args:
        down_sample_point ((FPS_Num, 3), float64): downsampled points
        edge_points_label_bin ((FPS_Num, 1), int8): boolean array of edge points indicies
        vertices ((N, 3), float64): vertices
        edge_points_ori ((N,), int): indicies of original edge points from feature file(curves).
    """    
    down_sample_point[np.where(edge_points_label_bin == 1)[0], ]
    vertices[edge_points_ori, :]

def label_creator(num_points, nearest_neighbor_idx):
    """labels the sampled points based on nearest_neighbor_idx

    Args:
        num_points (int): size of sampled points
        nearest_neighbor_idx (int array): indices of edge points(corner points) in sampled points

    Returns:
        (num_points, 1): array of labels
    """
    if nearest_neighbor_idx is not None:
        PC_points_label_bin = np.zeros((num_points, 1), dtype = np.int8)
        PC_points_label_bin[nearest_neighbor_idx] += 1
        return PC_points_label_bin
    else:
        return np.zeros((num_points, 1), dtype = np.int8)

def nearest_neighbor_finder(edge_or_corner_points, down_sample_point):
    """finds a nearest neighbor from point of edge_or_corner_points in down_sample_point and return their indicies in
        down_sample_point.

    Args:
        edge_points ((edge_points_num, 3), np.array): edge_points coordinates in original model
        down_sample_point (FPS_num, 3): sampled points from faces.
    """
    if edge_or_corner_points.shape[0] == 0: # no points avilable
        return None
    return(np.argmin((np.apply_along_axis(np.subtract, 1, edge_or_corner_points, down_sample_point)**2).sum(axis = 2), axis=1))
    

def calc_distances(p0, points):
    """calculate Euclidean distance between points
    Code by Graipher from https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        p0 (float): anchor points
        points (float): neighbor pointspoints

    Returns:
        (float): Euclidean Distance(s)
    """
    """
    if len(p0.shape) == len(points.shape) == 1: # Distance point A <-> point B
        return ((p0 - points)**2).sum(axis = 0)
    elif len(p0.shape) == 1 and len(points.shape) == 2:
        return ((p0 - points)**2).sum(axis=1)
    """
    # remember that 
    return np.sqrt(((p0 - points)**2).sum(axis = len(points.shape) - 1))

#@timeit
def graipher_FPS(pts, K):
    """ returns "greedy" farthest points based on Euclidean Distances
    Code by Graipher from https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        pts ((N, 3), float): Sampled points
        K (int): the number of farthest sampled points

    Returns:
        farthest_pts ((K, 3), float): farthest sampled points
    """
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in tqdm(range(1, K)):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def sharp_edges(list_ftr_line):
    """returns true if the model contains at least one sharp edges type of Circle, BSpline or Line.

    Args:
        list_stt_file (str): filename of .yml stats file

    Returns:
        bool
    """    
    f = open(list_ftr_line, "r")
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line[:-1] == "  sharp: true": # true detected.
            if lines[idx+1][8:-1] in ["Circle", "BSpline", "Line"]:
                f.close()
                return True
    f.close()
    return False

def delete_newline(line):
    if line[-1:] == "\n": line = line[:-1]
    return line

def delete_spaces(line):
    if line[:4] == "    ": line = line[4:]
    return line


def cross_points_finder(curve, other_curve):
    """[summary]

    Args:
        curve ([type]): [description]
        other_curve ([type]): [description]

    Returns:
        [type]: [description]
    """    
    cross_points = []
    for i in curve[1:-1]:
        for j in other_curve[1:-1]:
            if i == j:
                cross_points.append(i)
    return cross_points


def another_half_curve_pair_exist(curve, all_curves, circle_pair_index):
    """[summary]

    Args:
        curve ([type]): [description]
        all_curves ([type]): [description]
        circle_pair_index ([type]): [description]

    Returns:
        [type]: [description]
    """    
    k = 0        
    for candidate in all_curves:
        # check if it's same type
        # cross check the end points
        if curve[0] == candidate[0] and curve[1][0] == candidate[1][-1] and curve[1][-1] == candidate[1][0]:
            circle_pair_index[0] = k # index update
            return True
        k = k + 1
    return False

def update_lists(curve, open_curves, corner_points_ori, edge_points_ori):
    """[summary]

    Args:
        curve ([type]): [description]
        open_curves ([type]): [description]
        corner_points_ori ([type]): [description]
        edge_points_ori ([type]): [description]
    """    
    '''
    open_curves.append(curve)
    corner_points_ori.append(curve[1][0])
    corner_points_ori.append(curve[1][-1])
    edge_points_ori =  edge_points_ori + curve[1][:]
    '''
    open_curves.append(curve)
    corner_points_ori = corner_points_ori+[curve[1][0], curve[1][-1]]
    edge_points_ori = edge_points_ori+curve[1][:]
    return open_curves, corner_points_ori, edge_points_ori


def curves_with_vertex_indices(list_ftr_line):
    """ returns sharp curves with vertex indices. 

    Args:
        list_ftr_line (str): filename of features, .yml.

    Returns:
        curves: a list of lists of curves in this form: [['Circle', 0, 38, 49, ... 0], ['BSpline', 19, 299, 388, ..]]
    """

    f = open(list_ftr_line, "r")
    lines = f.readlines()
    in_curve_section = False
    curves = []
    #curves
    #surfaces:

    for idx, line in enumerate(lines):

        if line[:7] == "curves:": 
            in_curve_section = True
        elif (line[:-1] == "  sharp: true" or line[:-1] == "  sharp: false") and in_curve_section:
            curve = []
            if lines[idx+1][8:-1] in ["Circle", "BSpline", "Line"]:

                # append name of curve type
                curve.append(lines[idx+1][8:-1])

                # get ready to append the vert_indices
                string = ''
                vert_indices_line_idx = idx+2
                open_bracket_idx = lines[vert_indices_line_idx].find("[")

                # Check if it is a line or multiple lines
                if "]" in lines[vert_indices_line_idx]: # closed bracket in the same line
                    closed_bracket_idx = lines[vert_indices_line_idx].find("]")
                    string = string + lines[vert_indices_line_idx][open_bracket_idx:closed_bracket_idx+1]
                    
                    curve.append(ast.literal_eval(string))
                    curves.append(curve)
                else: # multiple lines

                    # take the first line.
                    first_line = lines[vert_indices_line_idx][open_bracket_idx:][:]
                    first_line = delete_newline(first_line)
                    string = string + first_line
                    vert_indices_line_idx = vert_indices_line_idx + 1

                    # loop until there's a closed bracket.
                    while "]" not in lines[vert_indices_line_idx]:
                        nextline = lines[vert_indices_line_idx]
                        nextline = delete_newline(nextline)
                        nextline = delete_spaces(nextline)
                        string = string + nextline
                        vert_indices_line_idx = vert_indices_line_idx + 1

                    # add the last line of vert_indices
                    closed_bracket_idx = lines[vert_indices_line_idx].find("]")
                    lastline = lines[vert_indices_line_idx]
                    lastline = delete_spaces(lastline)
                    string = string + lastline[:closed_bracket_idx-3]
                    curve.append(ast.literal_eval(string))
                    curves.append(curve)
        elif line[:9] == "surfaces:": # text file reached surfaces. returns.
            return curves