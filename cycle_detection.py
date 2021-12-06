# Code mostly from https://www.geeksforgeeks.org/print-all-the-cycles-in-an-undirected-graph/
import numpy as np

class Cycle_Detector_in_BSplines:
    
    def __init__(self, BSpline_list):
        self.N = 100000
        self.BSpline_list = BSpline_list
        self.starting_vertices = [spline[2][0] for spline in self.BSpline_list]

    def run_cycle_detection_in_BSplines(self):
        for starting_vertex in self.starting_vertices:
            result_list = self.cycle_detection_in_BSplines(starting_vertex+1)
            
            if len(result_list) > 0:
                print("cycle_detection_in_BSplines: ", result_list)
                return True
            else:
                continue
        return False

    def cycle_detection_in_BSplines(self, starting_vertex):
        self.graph = [[] for i in range(self.N)]
        self.cycles = [[] for i in range(self.N)]
        BSpline_list_num = len(self.BSpline_list)
        max_num = 1
        min_num = 0
        for i in range(BSpline_list_num):
            # vertices in .yml file starts with idx 0. plus one.
            if max_num < np.max([self.BSpline_list[i][2][0]+1, self.BSpline_list[i][2][-1]+1]):
                max_num = np.max([self.BSpline_list[i][2][0]+1, self.BSpline_list[i][2][-1]+1])
            self.addEdge(self.BSpline_list[i][2][0]+1, self.BSpline_list[i][2][-1]+1)
            
        # arrays required to color the
        # graph, store the parent of node
        self.color = [0] * self.N
        self.par = [0] * self.N
    
        # mark with unique numbers
        self.mark = [0] * self.N
    
        # store the numbers of cycle
        self.cyclenumber = 0
        #edges = 13
    
        # call DFS to mark the cycles
        self.dfs_cycle(starting_vertex, 0, self.color, self.mark, self.par)
        
        self.cycle_list = []
        self.edges = max_num
        # function to print the cycles
        self.cycle_list = self.returnCycles(self.edges, self.mark, self.cycle_list)
        return self.cycle_list

    def dfs_cycle(self, u, p, color: list, mark: list, par: list):
    
        # already (completely) visited vertex.
        if color[u] == 2:
            return
    
        # seen vertex, but was not
        # completely visited -> cycle detected.
        # backtrack based on parents to
        # find the complete cycle.
        if color[u] == 1:
            self.cyclenumber += 1
            cur = p
            mark[cur] = self.cyclenumber
    
            # backtrack the vertex which are
            # in the current cycle thats found
            while cur != u:
                cur = par[cur]
                mark[cur] = self.cyclenumber
    
            return
    
        par[u] = p
    
        # partially visited.
        color[u] = 1
    
        # simple dfs on graph
        for v in self.graph[u]:
    
            # if it has not been visited previously
            if v == par[u]:
                continue
            self.dfs_cycle(v, u, color, mark, par)
    
        # completely visited.
        color[u] = 2
    
    # add the edges to the graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    # Function to print the cycles
    def returnCycles(self, edges, mark: list, cycle_list: list):
        
        # push the edges that into the
        # cycle adjacency list
        for i in range(1, edges + 1):
            if mark[i] != 0:
                self.cycles[mark[i]].append(i)
        
        # print all the vertex with same cycle
        for i in range(1, self.cyclenumber + 1):
            cycle_list.append(self.cycles[i])
            # Print the i-th cycle  
            print("Cycle Number %d:" % i, end = " ")
            for x in self.cycles[i]:
                print(x, end = " ")
            print()

        return cycle_list






'''
# Python3 program to print all the cycles
# in an undirected graph
N = 100000

# variables to be used
# in both functions
graph = [[] for i in range(N)]
cycles = [[] for i in range(N)]
 
 
# Function to mark the vertex with
# different colors for different cycles
def dfs_cycle(u, p, color: list, mark: list, par: list):
    global cyclenumber
 
    # already (completely) visited vertex.
    if color[u] == 2:
        return
 
    # seen vertex, but was not
    # completely visited -> cycle detected.
    # backtrack based on parents to
    # find the complete cycle.
    if color[u] == 1:
        cyclenumber += 1
        cur = p
        mark[cur] = cyclenumber
 
        # backtrack the vertex which are
        # in the current cycle thats found
        while cur != u:
            cur = par[cur]
            mark[cur] = cyclenumber
 
        return
 
    par[u] = p
 
    # partially visited.
    color[u] = 1
 
    # simple dfs on graph
    for v in graph[u]:
 
        # if it has not been visited previously
        if v == par[u]:
            continue
        dfs_cycle(v, u, color, mark, par)
 
    # completely visited.
    color[u] = 2
 
# add the edges to the graph
def addEdge(u, v):
    graph[u].append(v)
    graph[v].append(u)
 
# Function to print the cycles
def returnCycles(edges, mark: list, cycle_list: list):
    
    # push the edges that into the
    # cycle adjacency list
    for i in range(1, edges + 1):
        if mark[i] != 0:
            cycles[mark[i]].append(i)
    
    # print all the vertex with same cycle
    for i in range(1, cyclenumber + 1):
        cycle_list.append(cycles[i])
        # Print the i-th cycle  
        print("Cycle Number %d:" % i, end = " ")
        for x in cycles[i]:
            print(x, end = " ")
        print()

    return cycle_list

def cycle_detection_in_BSplines(BSpline_list):
    BSpline_list_num = len(BSpline_list)
    for i in range(BSpline_list_num):
        addEdge(BSpline_list[i][2][0], BSpline_list[i][2][-1])
    # arrays required to color the
    # graph, store the parent of node
    color = [0] * N
    par = [0] * N
 
    # mark with unique numbers
    mark = [0] * N
 
    # store the numbers of cycle
    global cyclenumber
    cyclenumber = 0
    edges = 13
 
    # call DFS to mark the cycles
    dfs_cycle(1, 0, color, mark, par)
    
    cycle_list = []
    # function to print the cycles
    cycle_list = returnCycles(edges, mark, cycle_list)
    return cycle_list

# Driver Code

if __name__ == "__main__":
 
    # add edges
    Cycle_Detector = Cycle_Detector_in_BSplines([])
    
    Cycle_Detector.addEdge(4, 5)
    Cycle_Detector.addEdge(5, 8)
    Cycle_Detector.addEdge(8, 9)
    Cycle_Detector.addEdge(9, 4)
    
    Cycle_Detector.addEdge(1, 2)
    Cycle_Detector.addEdge(2, 3)
    Cycle_Detector.addEdge(3, 4)
    Cycle_Detector.addEdge(4, 6)
    Cycle_Detector.addEdge(4, 7)
    Cycle_Detector.addEdge(5, 6)
    Cycle_Detector.addEdge(3, 5)
    Cycle_Detector.addEdge(7, 8)
    Cycle_Detector.addEdge(6, 10)
    Cycle_Detector.addEdge(5, 9)
    Cycle_Detector.addEdge(10, 11)
    Cycle_Detector.addEdge(11, 12)
    Cycle_Detector.addEdge(11, 13)
    Cycle_Detector.addEdge(12, 13)
    
 
    # arrays required to color the
    # graph, store the parent of node
    Cycle_Detector.color = [0] * Cycle_Detector.N
    Cycle_Detector.par = [0] * Cycle_Detector.N
 
    # mark with unique numbers
    Cycle_Detector.mark = [0] * Cycle_Detector.N
    Cycle_Detector.cyclenumber = 0
    # store the numbers of cycle
    #cyclenumber = 0
    Cycle_Detector.edges = 9
 
    # call DFS to mark the cycles
    Cycle_Detector.dfs_cycle(4, 0, Cycle_Detector.color, Cycle_Detector.mark, Cycle_Detector.par)
    
    # function to print the cycles
    cycle_list = []
    cycle_list = Cycle_Detector.returnCycles(Cycle_Detector.edges, Cycle_Detector.mark, cycle_list)
    print(cycle_list)
 
# This code is contributed by
# sanjeev2552
'''