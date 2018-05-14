# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow
python ./graphsage/graphsage/utils.py ${DATASPACE}/wikipedia/graphsage/POS-G.json ${DATASPACE}/wikipedia/graphsage/POS-walks.txt
# python ./graphsage/graphsage/unsupervised_train.py --train_prefix ${DATASPACE}/wikipedia/graphsage/POS --model graphsage_mean --max_total_steps 1000 --validate_iter 10
source activate base