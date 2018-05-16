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
    num_nodes = 2709
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    id_map = {}
    reverse_map = {}
    label_map = {}
    class_map = {}
    np.random.seed(1)
    random.seed(1)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    with open(folder + "/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            id_map[i] = i
            reverse_map[i] = info[0]
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            class_map[i] = labels[i].tolist()
            G.add_node(i, id=i, test=i in test, val=i in val, label=labels[i].tolist(), feature=feat_data[i].tolist())

    adj_lists = defaultdict(set)
    with open(folder + "/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            G.add_edge(paper1, paper2)
    data = json_graph.node_link_data(G)
    with open(folder + '/graphsage/ncora-G.json', 'w') as outfile:
        json.dump(data, outfile)
    with open(folder + '/graphsage/ncora-feats.npy', 'wb') as outfile:
        np.save(outfile, feat_data)
    with open(folder + '/graphsage/ncora-id_map.json', 'w') as outfile:
        json.dump(id_map, outfile)
    with open(folder + '/graphsage/ncora-reverse_map.json', 'w') as outfile:
        json.dump(reverse_map, outfile)
    with open(folder + '/graphsage/ncora-class_map.json', 'w') as outfile:
        json.dump(class_map, outfile)

    print(folder + "/graphsage/")
    return

def load_wiki(folder):
    mat_file = folder + "/POS.mat"
    data = sio.loadmat(mat_file)
    
    num_nodes = data['group'].shape[0]
    print(num_nodes)
    np.random.seed(1)
    random.seed(1)
    rand_indices = np.random.permutation(num_nodes)
    train = rand_indices[:int(num_nodes * 0.64)]
    val = rand_indices[int(num_nodes * 0.64):int(num_nodes * 0.8)]
    test = rand_indices[int(num_nodes * 0.8):]
    
    G = nx.Graph()
    class_map = {}
    id_map = {}
    classes = data['group'].todense()
    for i in range(num_nodes):
        id_map[i] = i
        class_map[i] = classes[i].tolist()[0]
        class_map[i] = next(i for i,v in enumerate(class_map[i]) if v > 0) #wikipedia dataset has only 1 label
        G.add_node(i, id=i, test=i in test, val=i in val)
    
    A = coo_matrix(data['network'])
    for i,j,v in zip(A.row, A.col, A.data):
        G.add_edge(int(i), int(j), weight=float(v))

    with open(folder + '/graphsage/POS-G.json', 'w') as outfile:
        json.dump(json_graph.node_link_data(G), outfile)
    # with open(folder + 'graphsage/POS-feats.npy', 'wb') as outfile:
    #     np.save(outfile, feat_data)
    with open(folder + '/graphsage/POS-id_map.json', 'w') as outfile:
        json.dump(id_map, outfile)
    # with open(folder + 'graphsage/POS-reverse_map.json', 'w') as outfile:
    #     json.dump(reverse_map, outfile)
    with open(folder + '/graphsage/POS-class_map.json', 'w') as outfile:
        json.dump(class_map, outfile)

    print(folder + "/graphsage/")
    return

def main(args):
    if args.wiki:
        load_wiki(args.wiki)
    if args.cora:
        load_cora(args.cora)
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Convert graph data to graphsage format.")
    parser.add_argument('--wiki', nargs='?', default='', help='Wikipedia data path')
    parser.add_argument('--cora', nargs='?', default='', help='Cora data path')
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