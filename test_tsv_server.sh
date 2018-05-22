DATASPACE=/mnt/storage01/duong/dataspace/graph

python data/export_tsv.py --algorithm node2vec --prefix facebook --emb ${DATASPACE}/facebook/emb/ --tsv ${DATASPACE}/facebook/tsv
python data/export_tsv.py --algorithm node2vec --prefix POS --emb ${DATASPACE}/wikipedia/emb/ --tsv ${DATASPACE}/wikipedia/tsv
python data/export_tsv.py --algorithm node2vec --prefix ca-astroph --emb ${DATASPACE}/ca-astroph/emb/ --tsv ${DATASPACE}/ca-astroph/tsv

python data/export_tsv.py --algorithm graphsage --prefix POS --emb ${DATASPACE}/wikipedia/unsup-graphsage/graphsage_mean_small_0.000010/ --tsv ${DATASPACE}/wikipedia/tsv
python data/export_tsv.py --algorithm graphsage --prefix ca-astroph --dataset ${DATASPACE}/ca-astroph/unsup-graphsage/graphsage_mean_small_0.000010/ --tsv ${DATASPACE}/ca-astroph/tsv