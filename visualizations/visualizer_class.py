import numpy as np
import open3d

class Visualizer_Prep:
        
    def generate_color_array1(self, arrayR, arrayB, arrayG, form):
        # color array for task 1 and 2
        color_array = np.zeros_like(form)
        colorR = [0.99, 0, 0.0] # red
        colorB = [0.0, 0.0, 0.99] # blue
        colorG = [0.0, 0.99, 0.0] # green
        color_array[0:arrayR.shape[0], :] = colorR
        color_array[np.where(arrayB == 1)[0], ] = colorB
        color_array[arrayR.shape[0]:, ] = colorG
        return color_array

    def update_visualize1(self):
        if self.rest_visualzations_num > 0 :
            arrayR = self.down_sample_point
            arrayB = self.point_label_list[self.next_visualization_num]
            arrayG = self.corrected_points_list[self.next_visualization_num]
            color_array = self.generate_color_array1(arrayR, arrayB, arrayG, form = np.concatenate([arrayR, arrayG], axis = 0))
            self.point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([arrayR, arrayG], axis = 0))
            self.point_cloud.colors = open3d.utility.Vector3dVector(color_array)
            self.vis.update_geometry()
            self.vis.update_renderer()
            self.vis.poll_events()
            self.vis.run()
            self.rest_visualzations_num = self.rest_visualzations_num - 1
            self.next_visualization_num = self.next_visualization_num + 1
        else:
            self.vis.destroy_window()

    def generate_corrected_points_list(self):
        self.corrected_points_list = [self.down_sample_point+self.edge_points_residual_vector, self.down_sample_point+self.corner_points_residual_vector]

    def generate_point_label_list(self):
        self.point_label_list  = [self.edge_points_label, self.corner_points_label]

    def visualize1(self):
        # Task1
        if self.task == 1:
            arrayR = self.down_sample_point
            arrayB = self.point_label_list[0]
            arrayG = self.corrected_points_list[0]

            color_array = self.generate_color_array1(arrayR, arrayB, arrayG, form = np.concatenate([arrayR, arrayG], axis = 0))
            self.point_cloud = open3d.geometry.PointCloud()
            self.point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([arrayR, arrayG], axis = 0))
            self.point_cloud.colors = open3d.utility.Vector3dVector(color_array)
            self.vis = open3d.visualization.Visualizer()
            self.vis.create_window(visible = False)
            self.vis.add_geometry(self.point_cloud)
            #self.vis.run()
            for i in range(1, 2):
                intput = input()
                if input == '':
                    arrayR = self.down_sample_point
                    arrayB = self.point_label_list[i]
                    arrayG = self.corrected_points_list[i]
                    color_array = self.generate_color_array1(arrayR, arrayB, arrayG, form = np.concatenate([arrayR, arrayG], axis = 0))
                    self.point_cloud.points = open3d.utility.Vector3dVector(np.concatenate([arrayR, arrayG], axis = 0))
                    self.point_cloud.colors = open3d.utility.Vector3dVector(color_array)
                    self.vis.update_geometry(self.point_cloud)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    #self.rest_visualzations_num = self.rest_visualzations_num - 1
                    #self.next_visualization_num = self.next_visualization_num + 1
                    #self.point_cloud.points = 
            self.vis.destroy_window()

            #self.vis = open3d.visualization.VisualizerWithKeyCallback()
            #self.vis.register_key_callback(257, update_visualize1)
            #self.vis.destroy_window()
            #for i in range(1, len(self.kwargs['point_label_list'])):
            #    self.update_visualize1(vis, point_cloud, self.kwargs, i)

            