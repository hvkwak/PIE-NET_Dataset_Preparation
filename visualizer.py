import open3d
import numpy as np
import scipy.io as sio
from scipy.interpolate import splprep, splev

def main():
    
    ref_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/5.mat')['Training_data']
    my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/0042_4.mat')['Training_data']

    ref_down_sample_point = ref_mat[0, 0]['down_sample_point'][0, 0]
    ref_PC_8096_edge_points_label_bin = np.where(ref_mat[0, 0]['PC_8096_edge_points_label_bin'][0, 0][:, 0] == 1)[0]
    ref_corner_points_label_bin = np.where(ref_mat[0, 0]['corner_points_label'][0, 0][:, 0] == 1)[0]
    view_point(ref_down_sample_point)
    view_point(ref_down_sample_point[ref_PC_8096_edge_points_label_bin, ])
    view_point(ref_down_sample_point[ref_corner_points_label_bin, ])

    my_down_sample_point = my_mat[0, 0]['down_sample_point'][0, 0]
    my_down_sample_point_edge = np.where(my_mat[0, 0]['edge_points_label'][0, 0][0,:] == 1)[0]
    my_down_sample_point_corner = np.where(my_mat[0, 0]['corner_points_label'][0, 0][0,:] == 1)[0]
    view_point(my_down_sample_point)
    view_point(my_down_sample_point[my_down_sample_point_edge, ])
    view_point(my_down_sample_point[my_down_sample_point_corner, ])
    
    '''
    from scipy.interpolate import splprep, splev
    x1 = np.array([0, 0, 0], dtype = np.float64)
    x2 = np.array([0, 0.3, 0.3], dtype = np.float64)
    x3 = np.array([0, 0.5, 1], dtype = np.float64)
    tck, u = splprep([x1, x2, x3], k = 2, s = 0)
    new_points = splev(u, tck)
    #points = np.concatenate((np.stack([x1, x2, x3]), np.stack(new_points)), axis = 0)
    view_point(new_points)
    
    x_new = np.zeros((42))
    y_new = np.zeros((42))
    phi = np.linspace(0, 2.*np.pi, 40)
    r = 0.5 + np.cos(phi)         # polar coords
    x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian
    x_new[0:40] = x
    y_new[0:40] = y
    x_new[40] = 2
    y_new[40] = 2
    x_new[41] = 2
    y_new[41] = 0
    from scipy.interpolate import splprep, splev
    tck, u = splprep([x_new, y_new], k = 1, s=0)
    new_points = splev(u, tck)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x_new, y_new, 'ro')
    ax.plot(new_points[0], new_points[1], 'r-')
    plt.show()
    
    phi = np.linspace(0, 2.*np.pi, 40)
    r = 0.5 + np.cos(phi)         # polar coords
    x, y = r * np.cos(phi), r * np.sin(phi)    # convert to cartesian
    from scipy.interpolate import splprep, splev
    tck, u = splprep([x, y], k = 0, s=1)
    new_points = splev(u, tck)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro')
    ax.plot(new_points[0], new_points[1], 'r-')
    plt.show()
    '''


def view_point(points):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.paint_uniform_color([0.0, 0.0, 0.0])
    point_cloud.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__": 
    main()