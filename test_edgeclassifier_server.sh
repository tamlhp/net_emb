# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow

# python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/facebook/ --embed_dir ${DATASPACE}/facebook/emb/ \
#     --prefix facebook --func hadamard --verbose --cache ${DATASPACE}/facebook/cache-node2vec/
# python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/facebook/ --embed_dir ${DATASPACE}/facebook/unsup-graphsage/gcn_big_0.000010/ \
#     --prefix facebook --func hadamard --verbose --cache ${DATASPACE}/facebook/cache-graphsage/

python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/wikipedia/ --embed_dir ${DATASPACE}/wikipedia/emb/ \
    --prefix POS --weighted --func hadamard --verbose --cache ${DATASPACE}/wikipedia/cache-node2vec/
python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/wikipedia/ --embed_dir ${DATASPACE}/wikipedia/unsup-graphsage/gcn_big_0.000010 \
    --prefix POS --weighted --func hadamard --verbose --cache ${DATASPACE}/wikipedia/cache-graphsage/

python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/blogcatalog/ --embed_dir ${DATASPACE}/blogcatalog/emb/ \
    --prefix blog --func hadamard --verbose --cache ${DATASPACE}/blogcatalog/cache-node2vec/
python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/blogcatalog/ --embed_dir ${DATASPACE}/blogcatalog/unsup-graphsage/gcn_big_0.000010 \
    --prefix blog --func hadamard --verbose --cache ${DATASPACE}/blogcatalog/cache-graphsage/

python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/ca-astroph/ --embed_dir ${DATASPACE}/ca-astroph/emb/ \
    --prefix ca-astroph --func hadamard --verbose --cache ${DATASPACE}/ca-astroph/cache-node2vec/
python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/ca-astroph/ --embed_dir ${DATASPACE}/ca-astroph/unsup-graphsage/gcn_big_0.000010 \
    --prefix ca-astroph --func hadamard --verbose --cache ${DATASPACE}/ca-astroph/cache-graphsage/

python ./edge2vec/edge_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/ppi/ --embed_dir ${DATASPACE}/ppi/emb/ \
    --prefix ppi --func hadamard --verbose --cache ${DATASPACE}/ppi/cache-node2vec/
python ./edge2vec/edge_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/ppi/ --embed_dir ${DATASPACE}/ppi/unsup-graphsage/gcn_big_0.000010 \
    --prefix ppi --func hadamard --verbose --cache ${DATASPACE}/ppi/cache-graphsage/

source activate base