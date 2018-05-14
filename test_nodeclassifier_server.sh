DATASPACE=/Users/tnguyen/dataspace/graph
# DATASPACE=/mnt/storage01/duong/dataspace/graph
# source activate tensorflow
python node_classification/node_classifier.py --dataset_dir ${DATASPACE}/wikipedia/graphsage/ --embed_dir ${DATASPACE}/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010 \
    --prefix POS --setting test
# python node_classification/node_classifier.py --dataset_dir ${DATASPACE}/ppi/graphsage/ --embed_dir feat \
    # --prefix ppi --setting test
# source activate base