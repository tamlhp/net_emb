DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate python2
# python ./node2vec/src/main.py --input ./node2vec/graph/karate.edgelist --output ./node2vec/emb/karate.emb
# python ./node2vec/src/main.py --input /dataspace/graph/ppi/edgelist/ppi-subgraph.edgelist --output dataspace/graph/ppi/emb/ppi-subgraph.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/ppi/edgelist/ppi-G.edgelist --output ${DATASPACE/ppi/emb/ppi-G.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/wikipedia/edgelist/POS.edgelist --output ${DATASPACE}/wikipedia/emb/POS.emb --weighted
# python ./node2vec/src/main.py --input ${DATASPACE}/ppi/edgelist/ppi.edgelist --output ${DATASPACE}/ppi/emb/ppi.emb
python ./node2vec/src/main.py --input ${DATASPACE}/reddit/edgelist/reddit.edgelist --output ${DATASPACE}/reddit/emb/reddit.emb
source activate base