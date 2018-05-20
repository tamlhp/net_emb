# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
# source activate python2
# python ./edge2vec/edge2vec.py --edgelist ${DATASPACE}/wikipedia/edgelist/POS.edgelist --nodeemb ${DATASPACE}/wikipedia/emb/POS.emb \
    # --output ${DATASPACE}/wikipedia/emb/POS-edge.emb --weighted --func avg

# python ./edge2vec/edge_classifier.py --edgelist ${DATASPACE}/wikipedia/edgelist/POS.edgelist --nodeemb ${DATASPACE}/wikipedia/emb/POS.emb \
#     --weighted --func hadamard --verbose

python ./edge2vec/edge_classifier.py --edgelist ${DATASPACE}/ca-astroph/edgelist/ca-astroph.edgelist --nodeemb ${DATASPACE}/ca-astroph/emb/ca-astroph.emb \
     --func hadamard --verbose --cache ${DATASPACE}/ca-astroph/cache-node2vec/
# source activate base