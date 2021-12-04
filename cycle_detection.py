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
def printCycles(edges, mark: list):
 
    # push the edges that into the
    # cycle adjacency list
    for i in range(1, edges + 1):
        if mark[i] != 0:
            cycles[mark[i]].append(i)
 
    # print all the vertex with same cycle
    for i in range(1, cyclenumber + 1):
 
        # Print the i-th cycle
        print("Cycle Number %d:" % i, end = " ")
        for x in cycles[i]:
            print(x, end = " ")
        print()
 
# Driver Code
if __name__ == "__main__":
 
    # add edges
    addEdge(1, 2)
    addEdge(2, 3)
    addEdge(3, 4)
    addEdge(4, 6)
    addEdge(4, 7)
    addEdge(5, 6)
    addEdge(3, 5)
    addEdge(7, 8)
    addEdge(6, 10)
    addEdge(5, 9)
    addEdge(10, 11)
    addEdge(11, 12)
    addEdge(11, 13)
    addEdge(12, 13)
 
    # arrays required to color the
    # graph, store the parent of node
    color = [0] * N
    par = [0] * N
 
    # mark with unique numbers
    mark = [0] * N
 
    # store the numbers of cycle
    cyclenumber = 0
    edges = 13
 
    # call DFS to mark the cycles
    dfs_cycle(1, 0, color, mark, par)
 
    # function to print the cycles
    printCycles(edges, mark)
 
# This code is contributed by
# sanjeev2552