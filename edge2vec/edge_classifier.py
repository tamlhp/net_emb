from __future__ import print_function, division
import numpy as np
import time
import argparse
import pdb

import scipy.sparse as sp
import networkx as nx

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

def mask_test_edges(adj, args, test_frac=.1, val_frac=.05):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if args.verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if args.verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if args.prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if args.prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if args.verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if args.verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    if args.verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if args.verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if args.verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if args.verbose == True:
        print('Done with train-test split!')
        print('')

    assert len(train_edges) > 0 and len(test_edges) > 0 and len(val_edges) > 0

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false

def avg(x,y):
    return (x+y) / 2

def hadamard(x,y):
    return np.multiply(x,y)

def l1(x,y, weight=1):
    return np.abs(x-y)

def l2(x,y, weight=1):
    return (x-y)**2

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

def load_embedding(embedding_path):
    count = 0
    num_nodes = 0
    dim_size = 0
    node2vec = {}
    with open(embedding_path) as f:
        for line in f:
            if count==0:
                num_nodes, dim_size = [int(val) for val in line.split()]
            else:
                data = line.split()
                node_id = int(data[0])
                vector = [float(val) for val in data[1:]]
                node2vec[node_id] = vector
            count += 1
    return node2vec

def load_edgelist(edge_list_path, args=None):
    if args is None or not args.weighted:
        G = nx.read_edgelist(edge_list_path)
    else:
        G = nx.read_edgelist(edge_list_path, data=(('weight', float),))
    return G

def main(args):
    node2vec = load_embedding(args.nodeemb)
    G = load_edgelist(args.edgelist, args)

    emb_list = []
    for node_index in G.nodes:
        node_emb = node2vec[int(node_index)]
        emb_list.append(node_emb)
    
    adj_sparse = nx.to_scipy_sparse_matrix(G)
    train_test_split = mask_test_edges(adj_sparse, args)

    funcs = {
        "avg" : avg,
        "hadamard" : hadamard,
        "l1" : l1,
        "l2" : l2,
    }

    args.func = funcs.get(args.func, hadamard)
    scores = edge_classify(emb_list,train_test_split, args)
    print(scores)
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Link prediction")
    parser.add_argument('--edgelist', default="/Users/tnguyen/dataspace/graph/wikipedia/edgelist/POS.edgelist", help='Edgelist file')
    parser.add_argument('--nodeemb', default="/Users/tnguyen/dataspace/graph/wikipedia/emb/POS.emb", help='Node embedding file')
    parser.add_argument('--weighted', action='store_true', default=False, help='Weighted or not')
    parser.add_argument('--prevent_disconnect', action='store_true', default=True, help='Edge discard strategy for link prediction')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose or not')
    parser.add_argument('--func', choices=['avg', 'hadamard','l1', 'l2'], default="hadamard", help='Binary operator')
    parser.add_argument('--edge_score_mode', choices=['edge-emb', 'dot-product'], default="edge-emb", help='Edge embedding choice')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
    