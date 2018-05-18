from __future__ import print_function
import json
import time
import numpy as np
import pdb
import sys
sys.path.insert(0, "./")
from edge2vec import edge2vec

import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser

def run_regression(train_embeds, train_labels, test_embeds, test_labels, args):
    start_time = time.time()

    np.random.seed(1)


    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier

    # dummy = DummyClassifier()
    # dummy.fit(train_embeds, train_labels)
    if args.label == 'single':
        log = SGDClassifier(loss="log", n_jobs=55)
    elif args.label == 'multi':
        log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
    else:
        assert False
    log.fit(train_embeds, train_labels)

    n2v_scores = {}

    if args.label == 'single':
        n2v_scores['test_f1'] = f1_score(test_labels, log.predict(test_embeds), average=args.average)
        n2v_scores['train_f1'] = f1_score(train_labels, log.predict(train_embeds), average=args.average)
        n2v_scores['runtime'] = time.time() - start_time

        print("Test F1-score", n2v_scores['test_f1'])
        print("Train F1-score", n2v_scores['train_f1'])
        print("Runtime (s)", n2v_scores['runtime'])
        # print("Random baseline")
        # print(f1_score(test_labels, dummy.predict(test_embeds), average=average))
    elif args.label == 'multi':
        for i in range(test_labels.shape[1]):
            print("F1 score", f1_score(test_labels[:,i], log.predict(test_embeds)[:,i], average="micro"))
        # for i in range(test_labels.shape[1]):
            # print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro")
    else:
        assert False

    return n2v_scores

def parse_args():
    parser = ArgumentParser("Node classification")
    parser.add_argument("--algorithm", choices=['node2vec', 'graphsage'], default="graphsage", help="network embedding method")
    parser.add_argument("--dataset_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/graphsage", help="Path to directory containing the dataset.")
    parser.add_argument("--embed_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("--prefix", default="POS", help="Prefix to access the dataset")
    parser.add_argument("--setting", choices=['val', 'test'], default="test", help="Either val or test.")
    parser.add_argument("--label", choices=['multi', 'single'], default="single", help="Either multi or single lalbel classification.")
    parser.add_argument("--average", choices=['micro', 'macro', 'weighted', 'none'], default="micro", help="Average strategy for multi-class classification")
    return parser.parse_args()

def main(args):
    dataset_dir = args.dataset_dir
    emb_dir = args.embed_dir
    setting = args.setting
    prefix = args.prefix
    if args.average == 'none': args.average=None

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open("{0}/{1}-G.json".format(dataset_dir, prefix))))
    print(nx.info(G))

    labels = json.load(open("{0}/{1}-class_map.json".format(dataset_dir, prefix)))
    # labels = {int(i):l for i, l in labels.iteritems()}
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]

    train_labels = np.array([labels[str(i)] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[str(i)] for i in test_ids])
    print("running", emb_dir)

    if emb_dir == "feat":
        print("Using only features..")
        feats = np.load("{0}/{1}-feats.npy".format(dataset_dir, prefix))
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:,0] = np.log(feats[:,0]+1.0)
        feats[:,1] = np.log(feats[:,1]-min(np.min(feats[:,1]), -1))
        feat_id_map = json.load(open("{0}/{1}-id_map.json".format(dataset_dir, prefix)))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]] 
        test_feats = feats[[feat_id_map[id] for id in test_ids]] 

        print("Running regression..")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        run_regression(train_feats, train_labels, test_feats, test_labels, args)
    else:
        if args.algorithm == 'graphsage':
            embeds = np.load(emb_dir + "/val.npy")
            id_map = {}
            with open(emb_dir + "/val.txt") as fp:
                for i, line in enumerate(fp):
                    id_map[line.strip()] = i
        elif args.algorithm == 'node2vec':
            node2vec, num_nodes, dim_size = edge2vec.load_embedding("{0}/{1}.emb".format(emb_dir, prefix))
            id_map = json.load(open("{0}/{1}-id_map.json".format(dataset_dir, prefix)))
            embeds = np.zeros(shape=(num_nodes, dim_size))
            for i in range(num_nodes):
                embeds[i] = node2vec[str(i)]
        else:
            assert False

        train_embeds = embeds[[id_map[str(id)] for id in train_ids]] 
        test_embeds = embeds[[id_map[str(id)] for id in test_ids]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels, args)
    return

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)