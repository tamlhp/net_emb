source activate python2
# python ./node2vec/src/main.py --input /dataspace/graphsage/ppi/edgelist/ppi-subgraph.edgelist --output dataspace/graphsage/ppi/emb/ppi-subgraph.emb
python ./node2vec/src/main.py --input /mnt/storage01/duong/dataspace/graphsage/ppi/edgelist/ppi-G.edgelist --output /mnt/storage01/duong/dataspace/graphsage/ppi/emb/ppi-G.emb
source activate base