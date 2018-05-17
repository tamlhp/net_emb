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

    print(nx.info(G))
    nx.write_edgelist(G, folder + '/edgelist/cora.edgelist')
    print(folder + "/edgelist/")
    return

def load_wiki(folder):
    mat_file = folder + "/POS.mat"
    data = sio.loadmat(mat_file)
    
    G = nx.Graph(data['network'])
    print(nx.info(G))
    nx.write_edgelist(G, path=folder + "/edgelist/POS.edgelist", delimiter=" ", data=['weight'])

    print(folder + "/edgelist/")
    return

def load_reddit(folder):
    G = json_graph.node_link_graph(json.load(open("{0}/graphsage/{1}-G.json".format(folder, "reddit"))))
    print(nx.info(G))
    nx.write_edgelist(G, path=folder + "/edgelist/reddit.edgelist", delimiter=" ", data=['weight'])
    print(folder + "/edgelist/")
    return

def load_ppi(folder):
    G = json_graph.node_link_graph(json.load(open("{0}/graphsage/{1}-G.json".format(folder, "ppi"))))
    print(nx.info(G))
    nx.write_edgelist(G, path=folder + "/edgelist/ppi.edgelist", delimiter=" ", data=['weight'])
    print(folder + "/edgelist/")
    return

def main(args):
    if args.wiki:
        load_wiki(args.wiki)
    if args.cora:
        load_cora(args.cora)
    if args.reddit:
        load_reddit(args.reddit)
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Convert graph data to edgelist format.")
    parser.add_argument('--wiki', nargs='?', default='', help='Wikipedia data path')
    parser.add_argument('--cora', nargs='?', default='', help='Cora data path')
    parser.add_argument('--reddit', nargs='?', default='', help='Reddit data path')
    parser.add_argument('--ppi', nargs='?', default='', help='PPI data path')
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