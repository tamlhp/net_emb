# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
# source activate python2
python ./edge2vec/edge_classifier_graphsage.py --dataset_dir ${DATASPACE}/wikipedia/graphsage --embed_dir ${DATASPACE}/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010 \
    --prefix POS --weighted --func hadamard --verbose
# source activate base