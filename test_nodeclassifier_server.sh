DATASPACE=/Users/tnguyen/dataspace/graph
# DATASPACE=/mnt/storage01/duong/dataspace/graph
# source activate tensorflow
# python node_classification/node_classifier.py --algorithm graphsage --dataset_dir ${DATASPACE}/wikipedia/graphsage/ \
#     --embed_dir ${DATASPACE}/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010 \
#     --prefix POS --setting test
# python node_classification/node_classifier.py --dataset_dir ${DATASPACE}/ppi/graphsage/ --embed_dir feat \
    # --prefix ppi --setting test
python node_classification/node_classifier.py --algorithm node2vec --dataset_dir ${DATASPACE}/wikipedia/graphsage/ \
    --embed_dir ${DATASPACE}/wikipedia/emb/ \
    --prefix POS --setting test
# source activate base