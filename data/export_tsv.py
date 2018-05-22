from __future__ import print_function, division
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert to tsv format.")
    parser.add_argument('--dataset', default="/Users/tnguyen/dataspace/graph/karate", help='Dataset path')
    parser.add_argument("--prefix", default="karate", help="Prefix to access the dataset")
    parser.add_argument('--algorithm', choices=['node2vec', 'graphsage'], default="node2vec", help='Network embedding algorithm')
    parser.add_argument('--tsv', default="/Users/tnguyen/dataspace/graph/karate/tsv", help='TSV folder path')
    return parser.parse_args()


def main(args):
    writer = open("{0}/{1}-{2}.tsv".format(args.tsv, args.prefix, args.algorithm), "wb")
    if args.algorithm == 'node2vec':   
        reader = open("{0}/emb/{1}.emb".format(args.dataset, args.prefix), "rb")
        reader.readline()
        for line in reader:
            data = line.split()
            writer.write("\t".join(data[1:]))
            writer.write("\n")
        
    elif args.algorithm == 'graphsage':
        embeds = np.load(args.dataset + "/unsup-karate/graphsage_mean_small_0.000010/val.npy")
        print(embeds.shape)
        for embed in embeds:
            writer.write(np.array2string(embed, separator="\t"))
            writer.write("\n")
    else:
        assert False

    writer.close()
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)