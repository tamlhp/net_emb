DATASPACE=/mnt/storage01/duong/dataspace/graph
python ./edge2vec/edge2vec.py --edgelist ${DATASPACE}/wikipedia/edgelist/POS.edgelist --nodeemb ${DATASPACE}/wikipedia/emb/POS.emb \,
    --output ${DATASPACE}/wikipedia/emb/POS-edge.emb --weighted --func avg