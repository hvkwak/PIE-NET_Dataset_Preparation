import sys
import numpy as np
from grafkom1Framework import ObjLoader

def main(args):

    """ generates the training dataset from ABC Dataset(Koch et al. 2019) suitable for the implementation of 
        PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020)

    Command Line Arguments:
        list_obj.txt ([string]): list of objects (.obj)
    
    Returns: A Python dictionary of training dataset.
    """

    print("args: ", args)
    list_obj_file = open(args[0], "r")
    list_obj_line = list_obj_file.readline()

    while list_obj_line:
        
        print(list_obj_line) # make sure that there's no "\n" in the line.
        if list_obj_line[-1:] == "\n": list_obj_line = list_obj_line[:-1]

        # load the object file
        Loader = ObjLoader(list_obj_line)
        vertices = Loader.vertices
        faces = Loader.faces

        print("vertices: ", vertices)
        print("faces: ", faces)

        list_obj_line = list_obj_file.readline()
    list_obj_file.close()



if __name__ == "__main__":
   main(sys.argv[1:]) # list_obj.txt, 