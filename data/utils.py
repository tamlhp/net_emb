import networkx as nx
import pdb

# function to calculate the number of triangles in a simple
# directed/undirected graph.
# isDirected is true if the graph is directed, its false otherwise
def countTriangle(G, isDirected):
    A = nx.adjacency_matrix(G)
    nodes = A.shape[0]
    count_Triangle = 0 #Initialize result
    # Consider every possible triplet of edges in graph
    for i in range(nodes):
        for j in range(nodes):
            for k in range(nodes):
                # check the triplet if it satisfies the condition
                if( i!=j and i !=k and j !=k and
                        A[i,j] and A[j,k] and A[k,i]):
                    count_Triangle += 1
    # if graph is directed , division is done by 3
    # else division by 6 is done
    return count_Triangle/3 if isDirected else count_Triangle/6

