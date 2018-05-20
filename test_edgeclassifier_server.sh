# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow

# python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/wikipedia/graphsage --embed_dir ${DATASPACE}/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010 \
#     --prefix POS --weighted --func hadamard --verbose --cache ${DATASPACE}/wikipedia/cache-graphsage/

# python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/wikipedia/ --embed_dir ${DATASPACE}/wikipedia/emb/ \
#     --weighted --func hadamard --verbose --cache ${DATASPACE}/wikipedia/cache-node2vec/

# python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/ca-astroph/graphsage --embed_dir ${DATASPACE}/ca-astroph/unsup-graphsage/graphsage_mean_small_0.000010 \
#     --prefix ca-astroph --func hadamard --verbose --cache ${DATASPACE}/ca-astroph/cache-graphsage/

python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/ca-astroph/ --embed_dir ${DATASPACE}/ca-astroph/emb/ \
     --prefix ca-astroph --func hadamard --verbose --cache ${DATASPACE}/ca-astroph/cache-node2vec/

source activate base