# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow
cd graphsage/
# python -m graphsage.utils ${DATASPACE}/wikipedia/graphsage/POS-G.json ${DATASPACE}/wikipedia/graphsage/POS-walks.txt
# python -m graphsage.unsupervised_train --train_prefix ${DATASPACE}/wikipedia/graphsage/POS --model graphsage_mean --max_total_steps 1000 --validate_iter 10 \
    #  --identity_dim 128 --base_log_dir ${DATASPACE}/wikipedia/

# python -m graphsage.unsupervised_train --train_prefix ${DATASPACE}/reddit/graphsage/reddit --model graphsage_mean --max_total_steps 1000 --validate_iter 10 \
#      --identity_dim 128 --base_log_dir ${DATASPACE}/reddit/

# python -m graphsage.unsupervised_train --train_prefix ${DATASPACE}/ppi/graphsage/ppi --model graphsage_mean --max_total_steps 1000 --validate_iter 10 \
#      --identity_dim 128 --base_log_dir ${DATASPACE}/ppi/

python -m graphsage.unsupervised_train --train_prefix ${DATASPACE}/karate/graphsage/karate --model graphsage_mean --max_total_steps 1000 --validate_iter 10 \
     --identity_dim 128 --base_log_dir ${DATASPACE}/karate/
cd ../
source activate base