from __future__ import print_function, division
import networkx as nx
import pdb
import sys
import numpy as np
import argparse
import csv

def load_edgelist(edge_list_path, args=None):
    if args is None or not args.weighted:
        G = nx.read_edgelist(edge_list_path)
    else:
        G = nx.read_edgelist(edge_list_path, data=(('weight', float),))
    return G

def load_embedding(emb_file, args=None):    
    count = 0
    num_nodes = 0
    dim_size = 0
    node2vec = dict()
    with open(emb_file) as f:
        for line in f:
            if count==0:
                num_nodes, dim_size = [int(val) for val in line.split()]
            else:
                data = line.split()
                node_id = data[0]
                vector = np.array(data[1:]).astype(np.float)
                node2vec[str(node_id)] = vector
            count += 1
    pdb.set_trace()
    return node2vec, num_nodes, dim_size

def edge_emb(G, node2vec, func, args=None):
    edge2vec = {}
    for u,v,a in G.edges(data=True):
        t = (u,v)
        edge2vec[t] = func(node2vec[str(u)], node2vec[str(v)])
    return edge2vec

def avg(x,y):
    return (x+y) / 2

def hadamard(x,y):
    return np.multiply(x,y)

def l1(x,y, weight=1):
    return np.abs(x-y)

def l2(x,y, weight=1):
    return (x-y)**2

def main(args):
    G = load_edgelist(args.edgelist, args)
    node2vec, num_nodes, dim_size = load_embedding(args.nodeemb)
    func = globals()[args.func]
    edge2vec = edge_emb(G, node2vec, func)
    with open(args.output, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([len(edge2vec), dim_size])
        for k in edge2vec:
            writer.writerow([k[0], k[1]] + edge2vec[k].tolist())
        f.close()
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Generate edge embeddings from node embeddings")
    parser.add_argument('--edgelist', default="/Users/tnguyen/dataspace/graph/wikipedia/edgelist/POS.edgelist", help='Edgelist file')
    parser.add_argument('--nodeemb', default="/Users/tnguyen/dataspace/graph/wikipedia/emb/POS.emb", help='Node embedding file')
    parser.add_argument('--output', default="/Users/tnguyen/dataspace/graph/wikipedia/emb/POS-edge.emb", help='Node embedding file')
    parser.add_argument('--weighted', action='store_true', default=False, help='Weighted or not')
    parser.add_argument('--func', choices=['avg', 'hadamard','l1', 'l2'], default="avg", help='Binary operator')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
