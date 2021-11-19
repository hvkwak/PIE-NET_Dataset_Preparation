import numpy as np


class ObjLoader(object):
    """ Class object that loads .obj file.
        Code from https://inareous.github.io/posts/opening-obj-using-py
    """    
    def __init__(self, fileName):
        
        self.vertices = []
        self.faces = []
        self.vertex_normals = []

        ##
        try:
            f = open(fileName, "r")
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)
                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    #vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)
                elif line[0] == "f":
                    face = []
                    i1 = line.find(" ") + 1
                    i2 = line.find(" ", i1) + 1
                    i3 = line.find(" ", i2) + 1
                    s1 = line.find("//", i1, i2)
                    s2 = line.find("//", i2, i3)
                    s3 = line.find("//", i3)

                    # Note: in the object file vertices are 1-indexed per default: -1 !
                    index1 = int(line[i1:s1]) -1
                    index2 = int(line[i2:s2]) -1
                    index3 = int(line[i3:s3]) -1
                    '''
                    string = line.replace("//", "/")
                
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    '''
                    self.faces.append(tuple([index1, index2, index3]))
                elif line[:2] == "vn":                    
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex_normals = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    #vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertex_normals.append(vertex_normals)                    
            f.close()            
        except IOError:
            print(".obj file not found.")