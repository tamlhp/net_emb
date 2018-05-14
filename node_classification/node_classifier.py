from __future__ import print_function
import json
import numpy as np
import pdb

from networkx.readwrite import json_graph
from argparse import ArgumentParser

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    # dummy = DummyClassifier()
    # dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=55)
    log.fit(train_embeds, train_labels)
    print("Test F1-score")
    print(f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Train F1-score")
    print(f1_score(train_labels, log.predict(train_embeds), average="micro"))
    # print("Random baseline")
    # print(f1_score(test_labels, dummy.predict(test_embeds), average="micro"))

def parse_args():
    parser = ArgumentParser("Run evaluation on a dataset.")
    parser.add_argument("--dataset_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/graphsage", help="Path to directory containing the dataset.")
    parser.add_argument("--embed_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("--prefix", default="POS", help="Prefix to access the dataset")
    parser.add_argument("--setting", choices=['val', 'test'], default="test", help="Either val or test.")
    return parser.parse_args()

def main(args):
    dataset_dir = args.dataset_dir
    data_dir = args.embed_dir
    setting = args.setting
    prefix = args.prefix

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open("{0}/{1}-G.json".format(dataset_dir, prefix))))
    labels = json.load(open("{0}/{1}-class_map.json".format(dataset_dir, prefix)))
    labels = {int(i):l for i, l in labels.iteritems()}
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    train_labels = np.array([labels[i] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[i] for i in test_ids])
    print("running", data_dir)

    if data_dir == "feat":
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
        run_regression(train_feats, train_labels, test_feats, test_labels)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]] 
        test_embeds = embeds[[id_map[id] for id in test_ids]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
    return

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)