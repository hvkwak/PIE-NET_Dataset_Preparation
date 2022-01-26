import ast
import numpy as np
import itertools
from LongestPath import Graph
from tqdm import tqdm
#import open3d


def PathLength(Circle_list, nodes):
    max_digit = 0
    for circle in Circle_list:
        if max_digit < np.max([circle[2][0], circle[2][-1]]): max_digit = np.max([circle[2][0], circle[2][-1]])
    
    CurvesConnector = Graph(max_digit+1)
    for k in range(len(Circle_list)):
        CurvesConnector.addEdge(Circle_list[k][2][0], Circle_list[k][2][-1])

    node, node_2, LongDis = CurvesConnector.LongestPathLength()
    nodes[0] = node
    nodes[1] = node_2
    return LongDis

def Check_Connect_Circles(nodes, vertices, HalfCircle_list, Circle_list, BSpline_list):
    # connect the circles and determine whether it is half circle or BSpline.

    # note that
    # nodes[0] is start node
    # nodes[1] is end node
    assert nodes[0] != nodes[1]
    nodes.sort()

    merged = [None, None, []]
    temp_idx_list = []
    start_node_found = False
    end_node_found = False
    while not start_node_found or not end_node_found:
        for k in range(len(Circle_list)):
            # found the first node
            if not start_node_found and nodes[0] == Circle_list[k][2][0] and nodes[1] != Circle_list[k][2][-1]:
                start_node_found = True
                merged[2] = merged[2] + Circle_list[k][2]
                nodes[0] = Circle_list[k][2][-1]
                temp_idx_list.append(k)
                #del Circle_list[k]
                break
            # next nodes
            elif start_node_found and nodes[0] == Circle_list[k][2][0] and nodes[1] != Circle_list[k][2][-1]:
                merged[2] = merged[2] + Circle_list[k][2][1:]
                nodes[0] = Circle_list[k][2][-1]
                temp_idx_list.append(k)
                #del Circle_list[k]
                break
            # final node
            elif start_node_found and nodes[0] == Circle_list[k][2][0] and nodes[1] == Circle_list[k][2][-1]:
                end_node_found = True
                merged[2] = merged[2] + Circle_list[k][2][1:]
                temp_idx_list.append(k)
                #del Circle_list[k]
                break

    # check wheter merged is half circle or BSpline
    point1_idx = merged[2][0]
    point2_idx = merged[2][-1]
    point1 = vertices[point1_idx, :]
    point2 = vertices[point2_idx, :]
    middle = (point1 + point2)/2.0
    distances = np.sqrt(((middle - vertices[merged[2][1:-1], :])**2).sum(axis = 1))
    small_std = np.std(distances, dtype = np.float64) < 0.025
    if len(merged[2]) > 3 and small_std:
        merged[0] = "HalfCircle"
        merged[1] = True
        HalfCircle_list.append(merged)
        Circle_list = [i for j, i in enumerate(Circle_list) if j not in temp_idx_list]
    else:
        for curve_idx in temp_idx_list:
            Circle_list[curve_idx][0] = "BSpline"
            BSpline_list.append(Circle_list[curve_idx])
        Circle_list = [i for j, i in enumerate(Circle_list) if j not in temp_idx_list]

    nodes[0] = None
    nodes[1] = None
    return HalfCircle_list, Circle_list




'''
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
'''

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
'''
def greedy_nearest_neighbor_finder(edge_or_corner_points, down_sample_point):
    """finds a nearest neighbor from point of edge_or_corner_points in down_sample_point and return their indicies in
        down_sample_point.

    Args:
        edge_points ((edge_points_num, 3), np.array): edge_points coordinates in original model
        down_sample_point (FPS_num, 3): sampled points from faces.
    """
    pts_num = edge_or_corner_points.shape[0]
    if pts_num == 0: # no points avilable
        return None
    pts_array = np.arange(pts_num)
    np.random.shuffle(pts_array)
    distances = (np.apply_along_axis(np.subtract, 1, edge_or_corner_points, down_sample_point)**2).sum(axis = 2)
    i = 0
    checked = []
    memory_arr = [None]*pts_num
    for i in pts_array:
        memory_arr[i] = np.argmin(distances[i, :])
        distances[:, memory_arr[i]] = np.Inf
        i = i+1
    return np.array(memory_arr)
'''    
    
def nearest_neighbor_finder(edge_or_corner_points, down_sample_point, use_clustering, neighbor_distance):
    
    pts_num = edge_or_corner_points.shape[0]
    down_sample_pts_num = down_sample_point.shape[0]
    if pts_num == 0: # no points avilable
        return None

    # "empty" arrays to check if a neighbor point was assigned to each point of edge_or_corner_points.
    edge_or_corner_points_neighbor = np.full((pts_num, ), -1) # first start with an array full of -1 and update it.
    down_sample_pts_used = np.zeros((down_sample_pts_num, ), dtype=np.int8) # array of Falses

    # First compute distances and argmins
    distances = (np.apply_along_axis(np.subtract, 1, edge_or_corner_points, down_sample_point)**2).sum(axis = 2)
    argmin_per_row = np.argmin(distances, axis=1)

    # (Optional) what about second_argmins?
    """
    for i in range(distances.shape[0]):
        if distances[i, np.argpartition(distances[i, ], 2)[2]] - distances[i, argmin_per_row[i]] < 0.05:
            print("difference between first_argmin and second_argmin smaller than 0.05. Use the second_argmin")
            argmin_per_row[i] = np.argpartition(distances[i, ], 2)[2]
    """

    # Note that this will provide no optimal solution. But hopefully not too bad.
    # 1. This will generate duplicates of neighbors! But first take nearest neighbor anyway.
    unique, counts = np.unique(argmin_per_row, return_counts=True) 
    points_idx_with_one_neighbor = unique[counts == 1] # these are points indicies
    down_sample_pts_used[points_idx_with_one_neighbor] = 1 # these indicies are now True.
    for i in points_idx_with_one_neighbor:
        edge_or_corner_points_neighbor[np.where(argmin_per_row == i)[0][0]] = i # update with indicies
    

    # 2. Duplicates should be managed in their neighborhood with available points.
    points_idx_with_more_neighbors = unique[counts > 1] # point indicies in down_sampled_pts which has two or more neighbors in edge points
    if use_clustering:
        #
        # 2.1 (Optional) Clusering starts here: use a copied distance matrix.
        #
        #neighbor_distance = neighbor_distance
        bins = []
        excluded_items = []
        temp_down_sample_point = np.copy(down_sample_point)
        down_sample_points_distance_mat = (np.apply_along_axis(np.subtract, 1, temp_down_sample_point, temp_down_sample_point)**2).sum(axis = 2)
        
        # 2.2 filter the unavailable points out.
        mask = np.ones(down_sample_points_distance_mat.shape, dtype=bool)
        for i in points_idx_with_more_neighbors:
            mask[i, points_idx_with_more_neighbors] = False
        down_sample_points_distance_mat[mask] = np.Inf

        # 2.3 Start building bins with bigger bins.
        below_radius = down_sample_points_distance_mat < neighbor_distance
        row_sums = below_radius.sum(axis = 0)
        max_size = np.max(row_sums)
        while max_size > 0:
            array_nums = np.where(row_sums == max_size)[0] # take all the bins size of max_size(neighbors).
            print("array_nums of ", "max_size ", max_size, ":", len(array_nums))
            if max_size > 1:
                for k in array_nums:
                    bin_ = []
                    candidates = list(np.where((below_radius)[k, ])[0])
                    for i in range(len(candidates)):
                        if candidates[i] not in excluded_items:
                            bin_.append(candidates[i])
                            excluded_items.append(candidates[i])
                        '''
                        else:
                            print(candidates[i], " is in excluded_items. ", excluded_items)
                        '''
                    if len(bin_) > 0: bins.append(bin_)
            elif max_size == 1:
                for k in array_nums:
                    bins.append([k])
            max_size = max_size -1
        points_idx_with_more_neighbors = bins
        
    # clustering(making bins) ends here
    #
    for idx in points_idx_with_more_neighbors:
        current_edge_points = []
        if use_clustering:
            for k in idx:
                current_edge_points = current_edge_points + list(np.where(argmin_per_row == k)[0])
        else: 
            current_edge_points = np.where(argmin_per_row == idx)[0]
        current_edge_points_num = len(current_edge_points)            
        # generate permutations of available_neighbors.
        available_neighbors = available_neighbor_search(idx, down_sample_pts_used, down_sample_point, unique, counts, clustered = use_clustering)
        assert len(available_neighbors) == current_edge_points_num
        if current_edge_points_num > 7: 
            print("WARNING: too much Permutations. skip this.")
            raise ValueError("WARNING: too much Permutations. skip this.")
        perm = np.array(list(itertools.permutations(available_neighbors, current_edge_points_num)))
        """
        if perm.shape[0] > 50000: 
            print("WARNING: perm.shape[0] is ", perm.shape[0], "skip this.")
            raise ValueError("perm.shape[0] > 50000.")
        """
        #print("available_neighbors_num: ", current_edge_points_num, " this is ", perm.shape[0], " permutations.")
        best_distance = np.Inf
        best_perm = None
        # search for good permutation of indicies, where the total distance is minimal. this one is brute force.
        for k in range(perm.shape[0]):
            if distances[current_edge_points, perm[k]].sum(axis = 0) <  best_distance:
                best_distance = distances[current_edge_points, perm[k]].sum(axis = 0)
                best_perm = perm[k]
        # update the lists
        edge_or_corner_points_neighbor[current_edge_points] = best_perm
        down_sample_pts_used[best_perm] = 1
    
    # final check if it was successful.
    assert np.sum(edge_or_corner_points_neighbor == -1) == 0 and \
        np.unique(edge_or_corner_points_neighbor).shape[0] == np.sum(down_sample_pts_used == 1) == pts_num
        
    return edge_or_corner_points_neighbor

def available_neighbor_search(idx, down_sample_pts_used, down_sample_point, unique, counts, clustered):

    temp_array = np.copy(down_sample_point)
    unavailable_idx = np.where(down_sample_pts_used == 1)[0]
    temp_array[unavailable_idx, :] = np.Inf # these points were already assigned/unavailable.
    distances_to_neighbors, needed_available_neighbors = 0, 0

    if clustered:
        if len(idx) > 1: # clustered. note that they are multiple indicies
            distances_to_neighbors = ((np.mean(down_sample_point[idx, :], axis = 0) - temp_array)**2).sum(axis = 1)
            #distances_to_neighbors = (np.apply_along_axis(np.subtract, 1, down_sample_point[idx, :], temp_array)**2).sum(axis = 2)
            for k in range(len(idx)):
                needed_available_neighbors = needed_available_neighbors + counts[np.where(unique == idx[k])[0][0]]

        elif len(idx) == 1: # one idx at a time.
            distances_to_neighbors = ((down_sample_point[idx, :] - temp_array)**2).sum(axis = 1)
            needed_available_neighbors = counts[np.where(unique == idx[0])[0][0]]
    else:
        distances_to_neighbors = ((down_sample_point[idx, :] - temp_array)**2).sum(axis = 1)
        needed_available_neighbors = counts[np.where(unique == idx)[0][0]]

    ''' this was a bad idea.
    # grid-search: increase the size of radius to find possible neighbors in down_sample_point
    np.argpartition(distances[i, ], 2)[2]
    while needed_available_neighbors != np.sum(distances_to_neighbors < radius):
        # increase/decrease the radius accordingly.
        if needed_available_neighbors > np.sum(distances_to_neighbors < radius):
            radius = radius*1.3
        else:
            radius = radius*0.7
        i = i + 1
    #print("After ", i, " iterations it was successful to find a suitable radius: ", radius)
    available_neighbors = np.where(distances_to_neighbors < radius)
    '''
    return [np.argpartition(distances_to_neighbors, i)[i] for i in range(needed_available_neighbors)]

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
    return np.sqrt(((p0 - points)**2).sum(axis = len(points.shape) - 1))


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
    """returns points that meet somewhere BUT not in start/end points.
    """    
    cross_points = []
    for i in curve[1:-1]:
        for j in other_curve[1:-1]:
            if i == j:
                cross_points.append(i)
    return cross_points


def another_half_curve_pair_exist(curve, all_curves, circle_pair_index, circle_pair_Forward):
    """returns true if there are another half curve pair to form a complete circle.
    """    
    k = 0        
    for candidate in all_curves:
        # check if it's same type
        # cross check the end points
        if curve[0] == candidate[0] and (curve[2][0] == candidate[2][-1] and curve[2][-1] == candidate[2][0]):
            circle_pair_index[0] = k # index update
            circle_pair_Forward[0] = True
            return True
        elif curve[0] == candidate[0] and (curve[2][0] == candidate[2][0] and curve[2][-1] == candidate[2][-1]):
            circle_pair_index[0] = k # index update
            circle_pair_Forward[0] = False
            return True
        k = k + 1
    return False

def update_lists_open(curve, open_curves, corner_points_ori, edge_points_ori):
    """updates open curve list
    """    
    open_curves.append(curve)
    corner_points_ori = corner_points_ori+[curve[2][0], curve[2][-1]]
    edge_points_ori = edge_points_ori+curve[2][:]
    return open_curves, corner_points_ori, edge_points_ori

def half_curves_finder(Circle_or_BSpline_list, vertices):
    k = 0
    Circle_or_BSpline_num = len(Circle_or_BSpline_list)
    while k < Circle_or_BSpline_num:
        curve = Circle_or_BSpline_list[k]
        circle_pair_index = [None]
        circle_pair_Forward = [None]
        if curve[2][0] != curve[2][-1] and another_half_curve_pair_exist(curve, Circle_or_BSpline_list[k+1:], circle_pair_index, circle_pair_Forward):
            # this one consist of a pair of possible two half-circle curves.
            idx = k+1+circle_pair_index[0]
            new_curve = merge_two_half_circles_or_BSpline(curve, idx, Circle_or_BSpline_list, circle_pair_Forward)
            if is_circle(new_curve, vertices):
                Circle_or_BSpline_list[k] = new_curve
                del Circle_or_BSpline_list[idx]
                Circle_or_BSpline_num = Circle_or_BSpline_num - 1
        k = k + 1
    return Circle_or_BSpline_list

def is_circle(new_curve, vertices):
    centroid = vertices[new_curve[2], :].mean(axis = 0)
    return np.std(np.sqrt(np.sum((centroid - vertices[new_curve[2], :])**2, axis = 1))) < 0.1

def merge_two_half_circles_or_BSpline(curve, index_in_all_curves, all_curves, circle_pair_Forward):
    new_all_curves = all_curves.copy()
    new_curve = curve.copy()
    if circle_pair_Forward[0]:
        new_curve[2] = new_curve[2] + new_all_curves[index_in_all_curves][2][1:]
    else:
        new_all_curves[index_in_all_curves][2].reverse()
        new_curve[2] = new_curve[2] + new_all_curves[index_in_all_curves][2][1:]
    new_curve[0] = 'Circle'
    return new_curve

def degrees_same(temp_Splines):
    start_degree = temp_Splines[0][1]
    for i in range(1, len(temp_Splines)):
        if temp_Splines[i][1] != start_degree:
            return False
    return True

def touch_in_circles_or_BSplines(curve_list, Circle_list):
    circles_num = len(Circle_list)
    for i in range(circles_num):
        for j in range(circles_num):
            if curve_list[0] in Circle_list[i][2] and curve_list[-1] in Circle_list[j][2] and i != j:
                return True
    return False
        
def touch_in_circle(Line_list, Circle_list):
    circles_num = len(Circle_list)
    for i in range(circles_num):
        if Line_list[0] in Circle_list[i][2] or Line_list[-1] in Circle_list[i][2]:
            return True
    return False
                

def update_lists_closed(curve, closed_curves, edge_points_ori):
    closed_curves.append(curve)
    edge_points_ori = edge_points_ori+curve[2][:]
    return closed_curves, edge_points_ori

def mostly_sharp_edges(list_ftr_line, threshold):
    """ returns True if this CAD Model mostly consists of sharp edges. 

    Args:
        list_ftr_line (str): filename of features, .yml.

    Returns:
        curves: a list of lists of curves in this form: [['Circle', 0, 38, 49, ... 0], ['BSpline', 19, 299, 388, ..]]
    """
    f = open(list_ftr_line, "r")
    lines = f.readlines()
    in_curve_section = False
        
    sharp_true_count = 0
    sharp_false_count = 0
    for _, line in enumerate(lines):
        if line[:7] == "curves:": 
            in_curve_section = True
        elif (line[:-1] == "  sharp: false") and in_curve_section:
            sharp_false_count = sharp_false_count + 1
        elif (line[:-1] == "  sharp: true") and in_curve_section:
            sharp_true_count = sharp_true_count + 1
        elif line[:9] == "surfaces:": # text file reached surfaces. returns.
            break
    if sharp_true_count/np.float32(sharp_true_count+sharp_false_count) > threshold:
        return True
    else:
        return False

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
    for idx, line in enumerate(lines):
        if line[:7] == "curves:": 
            in_curve_section = True
        #elif (line[:-1] == "  sharp: false") and in_curve_section:
        #    print("sharp: false")
        elif (line[:-1] == "  sharp: true") and in_curve_section:
            #print("sharp: true")
            curve = []
            if lines[idx+1][8:-1] in ["Circle", "BSpline", "Line"]:
                degree = None
                if lines[idx+1][8:-1] == "BSpline":
                    degree_idx_search = idx+1
                    while lines[degree_idx_search][2:8] != "degree":
                        degree_idx_search = degree_idx_search -1
                    degree = int(lines[degree_idx_search][10:-1])

                # append name of curve type
                curve.append(lines[idx+1][8:-1])

                # append the degree of curve
                curve.append(degree)

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
            else:
                raise ValueError("curve not Curve, Line or BSpline.")
        elif line[:9] == "surfaces:": # text file reached surfaces. returns.
            return curves
