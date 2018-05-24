from __future__ import print_function, division
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert to tsv format.")
    parser.add_argument('--emb', default="/Users/tnguyen/dataspace/graph/karate/emb/", help='Emb path')
    parser.add_argument("--prefix", default="karate", help="Prefix to access the file")
    parser.add_argument('--algorithm', choices=['node2vec', 'graphsage'], default="node2vec", help='Network embedding algorithm')
    parser.add_argument('--tsv', default="/Users/tnguyen/dataspace/graph/karate/tsv", help='TSV folder path')
    return parser.parse_args()


def main(args):
    writer = open("{0}/{1}-{2}.tsv".format(args.tsv, args.prefix, args.algorithm), "wt")
    if args.algorithm == 'node2vec':   
        reader = open("{0}/{1}.emb".format(args.emb, args.prefix), "rt")
        reader.readline()
        for line in reader:
            data = line.split()
            writer.write("\t".join(data[1:]))
            writer.write("\n")
        
    elif args.algorithm == 'graphsage':
        embeds = np.load(args.emb + "/val.npy")
        for embed in embeds:
            writer.write("\t".join(map(str, embed)))
            writer.write("\n")
    else:
        assert False

    writer.close()
    print("{0}/{1}-{2}.tsv".format(args.tsv, args.prefix, args.algorithm))
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)