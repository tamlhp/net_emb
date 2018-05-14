# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow
cd graphsage/
# python -m graphsage.utils ${DATASPACE}/wikipedia/graphsage/POS-G.json ${DATASPACE}/wikipedia/graphsage/POS-walks.txt
python -m graphsage.unsupervised_train --train_prefix ${DATASPACE}/wikipedia/graphsage/POS --model graphsage_mean --max_total_steps 1000 --validate_iter 10 --identity_dim 50
cd ../
source activate base