from __future__ import print_function, division
import json
import time
import random
import numpy as np
import pdb
import sys
sys.path.insert(0, "./")
from edge2vec import edge2vec

import networkx as nx
from networkx.readwrite import json_graph
from argparse import ArgumentParser
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import *

seed = 123
np.random.seed(seed)
random.seed(seed)

def run_regression(train_embeds, train_labels, test_embeds, test_labels, args):
    start_time = time.time()

    if args.label == 'single':
        log = args.classifier
    elif args.label == 'multi':
        log = MultiOutputClassifier(args.classifier, n_jobs=-1)
        # log = RandomForestClassifier(n_jobs = -1, random_state=seed)
        # log = MLPClassifier(random_state=seed)
    else:
        assert False

    log.fit(train_embeds, train_labels)
    test_pred = log.predict(test_embeds)
    train_pred = log.predict(train_embeds)
    test_score = log.predict_proba(test_embeds)
    train_score = log.predict_proba(train_embeds) 

    n2v_scores = {}
    n2v_scores['runtime'] = time.time() - start_time

    if args.label == 'single':
        print("Single-label")
        

        n2v_scores['test_f1'] = f1_score(test_labels, test_pred, average=args.average)
        n2v_scores['test_precision'] = precision_score(test_labels, test_pred, average=args.average)
        n2v_scores['test_recall'] = recall_score(test_labels, test_pred, average=args.average)
        n2v_scores['test_accuracy'] = accuracy_score(test_labels, test_pred)
        
        n2v_scores['train_f1'] = f1_score(train_labels, train_pred, average=args.average)
        n2v_scores['train_precision'] = precision_score(train_labels, train_pred, average=args.average)
        n2v_scores['train_recall'] = recall_score(train_labels, train_pred, average=args.average)
        n2v_scores['train_accuracy'] = accuracy_score(train_labels, train_pred)

        lb = LabelBinarizer()
        lb.fit(test_labels)
        lb.fit(train_labels)
        n2v_scores['test_auc'] = roc_auc_score(lb.transform(test_labels), test_score, average=args.average)
        n2v_scores['test_ap'] = average_precision_score(lb.transform(test_labels), test_score, average=args.average)
        n2v_scores['train_auc'] = roc_auc_score(lb.transform(train_labels), train_score, average=args.average)
        n2v_scores['train_ap'] = average_precision_score(lb.transform(train_labels), train_score, average=args.average)
    elif args.label == 'multi':
        print("Multi-label", test_labels.shape[1])
        assert test_labels.shape[1] == train_labels.shape[1]

        n2v_scores['test_f1'] = []
        n2v_scores['train_f1'] = []
        n2v_scores['test_precision'] = []
        n2v_scores['train_precision'] = []
        n2v_scores['test_recall'] = []
        n2v_scores['train_recall'] = []
        for i in range(test_labels.shape[1]):
            n2v_scores['test_f1'].append(f1_score(test_labels[:,i], test_pred[:,i], average=args.average))
            n2v_scores['test_precision'].append(precision_score(test_labels[:,i], test_pred[:,i], average=args.average))
            n2v_scores['test_recall'].append(recall_score(test_labels[:,i], test_pred[:,i], average=args.average))
            n2v_scores['train_f1'].append(f1_score(train_labels[:,i], train_pred[:,i], average=args.average))
            n2v_scores['train_precision'].append(precision_score(train_labels[:,i], train_pred[:,i], average=args.average))
            n2v_scores['train_recall'].append(recall_score(train_labels[:,i], train_pred[:,i], average=args.average))
        n2v_scores['test_f1'] = np.mean(n2v_scores['test_f1'])
        n2v_scores['test_precision'] = np.mean(n2v_scores['test_precision'])
        n2v_scores['test_recall'] = np.mean(n2v_scores['test_recall'])
        n2v_scores['train_f1'] = np.mean(n2v_scores['train_f1'])
        n2v_scores['train_precision'] = np.mean(n2v_scores['train_precision'])
        n2v_scores['train_recall'] = np.mean(n2v_scores['train_recall'])

        n2v_scores['test_accuracy'] = accuracy_score(test_labels, test_pred)
        n2v_scores['train_accuracy'] = accuracy_score(train_labels, train_pred)

        # https://github.com/scikit-learn/scikit-learn/issues/2451
        # n2v_scores['test_lrap'] = label_ranking_average_precision_score(test_labels, test_score)
        # n2v_scores['train_lrap'] = label_ranking_average_precision_score(train_labels, train_score)
        # n2v_scores['test_auc'] = roc_auc_score(test_labels, test_score, average=args.average)
        # n2v_scores['test_ap'] = average_precision_score(test_labels, test_score, average=args.average)
        # n2v_scores['train_auc'] = roc_auc_score(train_labels, train_score, average=args.average)
        # n2v_scores['train_ap'] = average_precision_score(train_labels, train_score, average=args.average)s
    else:
        assert False

    print(n2v_scores)
    # print("Test F1-score", n2v_scores['test_f1'])
    # print("Train F1-score", n2v_scores['train_f1'])
    # print("Runtime (s)", n2v_scores['runtime'])

    # from sklearn.dummy import DummyClassifier
    # dummy = DummyClassifier()
    # dummy.fit(train_embeds, train_labels)
    # print("Random baseline")
    # print(f1_score(test_labels, dummy.predict(test_embeds), average=average))
    # for i in range(test_labels.shape[1]):
        # print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro")
    return n2v_scores

def parse_args():
    parser = ArgumentParser("Node classification")
    parser.add_argument("--algorithm", choices=['node2vec', 'graphsage'], default="graphsage", help="network embedding method")
    parser.add_argument("--dataset_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/graphsage", help="Path to directory containing the dataset.")
    parser.add_argument("--embed_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("--prefix", default="POS", help="Prefix to access the dataset")
    parser.add_argument("--setting", choices=['val', 'test'], default="test", help="Either val or test.")
    parser.add_argument("--label", choices=['multi', 'single'], default="single", help="Either single-label or multi-label classification.")
    parser.add_argument("--loss", choices=['log', 'hinge'], default="log", help="Loss function")
    parser.add_argument("--classifier", choices=['sgd', 'logistic'], default="sgd", help="Base classifier")
    parser.add_argument("--average", choices=['micro', 'macro', 'weighted', 'none'], default="micro", help="Average strategy for multi-class classification")
    return parser.parse_args()

def main(args):
    classifier = {
        "sgd" : SGDClassifier(loss=args.loss, n_jobs=-1, random_state=seed, max_iter=1000, tol=1e-3),
        "logistic" : LogisticRegression(random_state=seed, n_jobs=1),
    }
    args.classifier = classifier[args.classifier]

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
    if args.label == "multi" and train_labels.ndim == 1:
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
        # feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
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
            node2vec, id_map, num_nodes, dim_size = edge2vec.load_embedding("{0}/{1}.emb".format(emb_dir, prefix))
            # id_map = json.load(open("{0}/{1}-id_map.json".format(dataset_dir, prefix)))
            embeds = np.zeros(shape=(num_nodes, dim_size))
            for id in node2vec:
                embeds[id_map[id]] = node2vec[id]
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