import networkx as nx
import pdb
import json
import argparse

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

def change_node_id(emb_file, id_file, out_file):
    id_map = json.load(open(id_file))
    inv_map = {id_map[k]: k for k, v in id_map}
    writer = open(out_file, "wt")

    count = 0
    with open(emb_file) as f:
        for line in f:
            if count==0:
                writer.write(line + "\n")
            else:
                data = line.split()
                data[0] = inv_map[int(data[0])]
                writer.write(" ".join(data))
                writer.write("\n")
            count += 1
    write.close()
    return

def main(args):
    change_node_id(args.emb, args.id, args.out)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Utils.")
    parser.add_argument('--emb', help='Emb path')
    parser.add_argument('--id', help="Id map file")
    parser.add_argument('--out', help='Output file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)