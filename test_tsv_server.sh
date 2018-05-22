DATASPACE=/mnt/storage01/duong/dataspace/graph

# python data/export_tsv.py --algorithm node2vec --prefix karate --dataset ${DATASPACE}/karate --tsv ${DATASPACE}/karate/tsv
# python data/export_tsv.py --algorithm graphsage --prefix karate --dataset ${DATASPACE}/karate --tsv ${DATASPACE}/karate/tsv
python data/export_tsv.py --algorithm node2vec --prefix POS --dataset ${DATASPACE}/wikipedia --tsv ${DATASPACE}/wikipedia/tsv
python data/export_tsv.py --algorithm graphsage --prefix POS --dataset ${DATASPACE}/wikipedia --tsv ${DATASPACE}/wikipedia/tsv