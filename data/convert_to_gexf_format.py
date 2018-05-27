from __future__ import print_function, division
import argparse
import networkx as nx

def main(args):
    if args.weighted:
        G = nx.read_edgelist(args.edgelist, data=(('weight',float),))
    else:
        G = nx.read_edgelist(args.edgelist)
    nx.write_gexf(G, args.gexf)

    print(nx.info(G))
    if args.stat:
        # print("Diameter: " + str(nx.diameter(G)))
        print("Avg. clustering coefficient: " + str(nx.average_clustering(G)))
        print("# Triangles: " + str(sum(nx.triangles(G).values()) / 3))
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Convert graph data to gexf format.")
    parser.add_argument('--edgelist', nargs='?', default='', help='Input data path')
    parser.add_argument('--gexf', nargs='?', default='', help='Output data path')
    parser.add_argument('--weighted', action='store_true', default=False, help='Weighted or not')
    parser.add_argument('--stat', action='store_true', default=False, help='Some statistics')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)