from __future__ import print_function, division
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Convert to tsv format.")
    parser.add_argument('--emb', nargs='1', help='Embedding file')
    parser.add_argument('--algorithm', choices=['node2vec', 'graphsage'], nargs='1', help='Network embedding algorithm')
    parser.add_argument('--tsv', nargs='1', help='TSV output file path')
    return parser.parse_args()


def main(args):
    if args.algorithm == 'node2vec':
        
    elif args.algorithm == 'graphsage':

    else:
        assert False

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)