from __future__ import print_function, division
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
from collections import defaultdict
import random
import scipy.io as sio
from scipy.sparse import coo_matrix
import argparse
import math
import sys
import pdb
import csv
import utils

def load_cora(folder):
    G = nx.Graph()
    node_map = {}

    with open(folder + "/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            node_map[info[0]] = i
            G.add_node(i, id=i)

    with open(folder + "/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            G.add_edge(paper1, paper2)

    nx.write_edgelist(G, folder + '/edgelist/cora.edgelist')
    print(folder + "/edgelist/")
    return G

def load_wiki(folder):
    mat_file = folder + "/POS.mat"
    data = sio.loadmat(mat_file)
    
    G = nx.Graph(data['network'])
    nx.write_edgelist(G, path=folder + "/edgelist/POS.edgelist", delimiter=" ", data=['weight'])

    print(folder + "/edgelist/")
    return G

def load_reddit(folder):
    G = json_graph.node_link_graph(json.load(open("{0}/graphsage/{1}-G.json".format(folder, "reddit"))))
    nx.write_edgelist(G, path=folder + "/edgelist/reddit.edgelist", delimiter=" ", data=['weight'])
    print(folder + "/edgelist/")
    return G

def load_ppi(folder):
    G = json_graph.node_link_graph(json.load(open("{0}/graphsage/{1}-G.json".format(folder, "ppi"))))
    nx.write_edgelist(G, path=folder + "/edgelist/ppi.edgelist", delimiter=" ", data=['weight'])
    print(folder + "/edgelist/")
    return G

def load_blog(folder):
    G = nx.read_edgelist(folder + "/edges.csv", delimiter=",")
    nx.write_edgelist(G, path=folder + "/edgelist/blog.edgelist", delimiter=" ", data=['weight'])
    print(folder + "/edgelist/")
    return G

def main(args):
    if args.wiki:
        G = load_wiki(args.wiki)
    if args.cora:
        G = load_cora(args.cora)
    if args.reddit:
        G = load_reddit(args.reddit)
    if args.ppi:
        G = load_ppi(args.ppi)
    if args.blog:
        G = load_blog(args.blog)

    print(nx.info(G))

    if args.stat:
        # print("Diameter: " + str(nx.diameter(G)))
        print("Avg. clustering coefficient: " + str(nx.average_clustering(G)))
        print("# Triangles: " + str(sum(nx.triangles(G).values()) / 3))
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Convert graph data to edgelist format.")
    parser.add_argument('--wiki', nargs='?', default='', help='Wikipedia data path')
    parser.add_argument('--cora', nargs='?', default='', help='Cora data path')
    parser.add_argument('--reddit', nargs='?', default='', help='Reddit data path')
    parser.add_argument('--ppi', nargs='?', default='', help='PPI data path')
    parser.add_argument('--blog', nargs='?', default='', help='BlogCatalog data path')
    parser.add_argument('--stat', action='store_true', default=False, help='Some statistics')
    return parser.parse_args()

def test1():
    from mock import patch
    testargs = ["prog", 
                "--wiki", "/Users/tnguyen/dataspace/graph/wikipedia/",
                ]
    with patch.object(sys, 'argv', testargs):
        args = parse_args()
    print(' '.join(testargs))
    return args

if __name__ == '__main__':
    if '--ntt' in sys.argv:
        args = test1()
    else:
        args = parse_args()
    main(args)