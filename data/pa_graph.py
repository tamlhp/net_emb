from __future__ import print_function, division
import random
import numpy as np
import networkx as nx
import argparse
import math
import json
from networkx.readwrite import json_graph
import pdb

seed = 123
random.seed(seed)
np.random.seed(seed)

# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html

def parse_args():
    parser = argparse.ArgumentParser(description="Barabasi-Albert preferential attachment graph generator.")
    parser.add_argument('--path', default="/Users/tnguyen/dataspace/graph/pa_graph/", help='Path to save dataset')
    parser.add_argument('-n', type=int, default=10000, help='Number of nodes')
    parser.add_argument('--stat', action='store_true', default=False, help='Some statistics')
    return parser.parse_args()

def main(args):
    n = args.n
    writer = open(args.path + "/prefix.txt", "wt")
    for m in [1, 2, 3, 5, 10]:
        prefix = "pa,n={0},m={1}".format(n,m)
        writer.write(prefix + "\n")

        G = nx.barabasi_albert_graph(n,m,seed=seed)
        print(nx.info(G))
        edgelist = "{0}/edgelist/{1}.edgelist".format(args.path, prefix)
        nx.write_edgelist(G, path=edgelist, delimiter=" ", data=False)

        num_nodes = len(G.nodes())
        rand_indices = np.random.permutation(num_nodes)
        train = rand_indices[:int(num_nodes * 0.9025)]
        val = rand_indices[int(num_nodes * 0.9025):int(num_nodes * 0.95)]
        test = rand_indices[int(num_nodes * 0.95):]

        id_map = {}
        for i, node in enumerate(G.nodes()):
            id_map[str(node)] = i

        res = json_graph.node_link_data(G)
        res['nodes'] = [
            {
                'id': str(node['id']),
                'val': False,
                'test': False,
            }
            for node in res['nodes']]
        res['links'] = [
            {
                'source': id_map[str(link['source'])],
                'target': id_map[str(link['target'])],
            }
            for link in res['links']]

        with open('{0}/graphsage/{1}-G.json'.format(args.path, prefix), 'w') as outfile:
            json.dump(res, outfile)
        with open('{0}/graphsage/{1}-id_map.json'.format(args.path, prefix), 'w') as outfile:
            json.dump(id_map, outfile)
        
        if args.stat:
            try:
                print("Diameter: " + str(nx.diameter(G)))
            except:
                print("Diameter: N/A (Graph is disconnected)")
            print("Avg. clustering coefficient: " + str(nx.average_clustering(G)))
            print("# Triangles: " + str(sum(nx.triangles(G).values()) / 3))

    writer.close()
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)