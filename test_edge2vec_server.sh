# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
# source activate python2
python ./edge2vec/edge2vec.py --edgelist ${DATASPACE}/wikipedia/edgelist/POS.edgelist --nodeemb ${DATASPACE}/wikipedia/emb/POS.emb \
    --output ${DATASPACE}/wikipedia/emb/POS-edge.emb --weighted --func avg
# source activate base