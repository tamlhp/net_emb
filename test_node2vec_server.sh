# DATASPACE=/Users/tnguyen/dataspace/graph/
DATASPACE=/mnt/storage01/duong/dataspace/graph/
source activate python2

# python ./node2vec/src/main.py --input ${DATASPACE}/facebook/edgelist/facebook.edgelist --output ${DATASPACE}/facebook/emb/facebook.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/wikipedia/edgelist/POS.edgelist --output ${DATASPACE}/wikipedia/emb/POS.emb --weighted
# python ./node2vec/src/main.py --input ${DATASPACE}/blogcatalog/edgelist/blog.edgelist --output ${DATASPACE}/blogcatalog/emb/blog.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/ca-astroph/edgelist/ca-astroph.edgelist --output ${DATASPACE}/ca-astroph/emb/ca-astroph.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/ppi/edgelist/ppi.edgelist --output ${DATASPACE}/ppi/emb/ppi.emb
python ./node2vec/src/main.py --input ${DATASPACE}/reddit/edgelist/reddit.edgelist --output ${DATASPACE}/reddit/emb/reddit.emb


# python ./node2vec/src/main.py --input ${DATASPACE}/karate/edgelist/karate.edgelist --output ${DATASPACE}/karate/emb/karate.emb
# nice -n 19 python ./node2vec/src/main.py --input ${DATASPACE}/reddit/edgelist/reddit.edgelist --output ${DATASPACE}/reddit/emb/reddit.emb
# nice -n 19 python ./node2vec/src/main.py --input ${DATASPACE}/ca-astroph/edgelist/ca-astroph.edgelist --output ${DATASPACE}/ca-astroph/emb/ca-astroph.emb
source activate base