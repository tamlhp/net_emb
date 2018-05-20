from __future__ import print_function, division
import numpy as np
import time
import argparse
import json
import pdb
import edge2vec
from edge_classifier import mask_test_edges

import scipy.sparse as sp
import networkx as nx
from networkx.readwrite import json_graph

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    '''
    Input: positive test/val edges, negative test/val edges, edge score matrix.    
    Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
    '''
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score

def edge_classify(emb_list, train_test_split, args):
    '''
    @edge_score_mode: Whether to use bootstrapped edge embeddings + LogReg (like in node2vec paper), 
        or simple dot-product (like in GAE paper) for edge scoring
    '''

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    emb_matrix = np.vstack(emb_list)

    if args.edge_score_mode == "edge-emb":
        def get_edge_embeddings(edge_list):
            ''' 
            Generate bootstrapped edge embeddings (as is done in node2vec paper).
            Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2.
            '''
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = args.func(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print("Invalid edge_score_mode! Either use edge-emb or dot-product.")

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores

def main(args):
    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open("{0}/{1}-G.json".format(args.dataset_dir, args.prefix))))
    
    node_ids = [n for n in G.nodes()]
    
    print("running", args.embed_dir)

    embeds = np.load(args.embed_dir + "/val.npy")
    id_map = {}
    with open(args.embed_dir + "/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[int(line.strip())] = i
    embeds = embeds[[id_map[id] for id in node_ids]] 

    emb_list = []
    for node_index in G.nodes:
        node_emb = embeds[str(node_index)]
        emb_list.append(node_emb)
    
    adj_sparse = nx.to_scipy_sparse_matrix(G)
    train_test_split = mask_test_edges(adj_sparse, args)

    funcs = {
        "avg" : edge2vec.avg,
        "hadamard" : edge2vec.hadamard,
        "l1" : edge2vec.l1,
        "l2" : edge2vec.l2,
    }

    args.func = funcs.get(args.func, edge2vec.hadamard)
    scores = edge_classify(emb_list,train_test_split, args)
    print(scores)
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction")
    parser.add_argument("--dataset_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/graphsage", help="Path to directory containing the dataset.")
    parser.add_argument("--embed_dir", default="/Users/tnguyen/dataspace/graph/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument('--cache', nargs='?', default='', help='Cache folder for train test split')
    parser.add_argument("--prefix", default="POS", help="Prefix to access the dataset")
    parser.add_argument('--weighted', action='store_true', default=False, help='Weighted or not')
    parser.add_argument('--prevent_disconnect', action='store_true', default=True, help='Edge discard strategy for link prediction')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose or not')
    parser.add_argument('--func', choices=['avg', 'hadamard','l1', 'l2'], default="hadamard", help='Binary operator')
    parser.add_argument('--edge_score_mode', choices=['edge-emb', 'dot-product'], default="edge-emb", help='Edge embedding choice')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
    