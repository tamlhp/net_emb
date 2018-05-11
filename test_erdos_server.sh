# DATASPACE=/mnt/storage01/duong/dataspace/graph
DATASPACE=/Users/tnguyen/dataspace/graph
n=10000
p=0.00015
graph=erdos,n=${n},p=${p}
source activate python2
python ./node2vec/src/main.py --input ${DATASPACE}/erdos/edgelist/${graph}.edgelist --output ${DATASPACE}/erdos/emb/${graph}.emb
source activate base