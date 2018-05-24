DATASPACE=/mnt/storage01/duong/dataspace/graph

# python data/export_tsv.py --algorithm node2vec --prefix facebook --emb ${DATASPACE}/facebook/emb/ --tsv ${DATASPACE}/facebook/tsv
# python data/export_tsv.py --algorithm graphsage --prefix facebook --emb ${DATASPACE}/facebook/unsup-graphsage/gcn_big_0.000010/ --tsv ${DATASPACE}/facebook/tsv

# python data/export_tsv.py --algorithm node2vec --prefix POS --emb ${DATASPACE}/wikipedia/emb/ --tsv ${DATASPACE}/wikipedia/tsv
# python data/export_tsv.py --algorithm graphsage --prefix POS --emb ${DATASPACE}/wikipedia/unsup-graphsage/gcn_big_0.000010/ --tsv ${DATASPACE}/wikipedia/tsv

# python data/export_tsv.py --algorithm node2vec --prefix blog --emb ${DATASPACE}/blogcatalog/emb/ --tsv ${DATASPACE}/blogcatalog/tsv
# python data/export_tsv.py --algorithm graphsage --prefix blog --emb ${DATASPACE}/blogcatalog/unsup-graphsage/gcn_big_0.000010/ --tsv ${DATASPACE}/blogcatalog/tsv

# python data/export_tsv.py --algorithm node2vec --prefix ca-astroph --emb ${DATASPACE}/ca-astroph/emb/ --tsv ${DATASPACE}/ca-astroph/tsv
# python data/export_tsv.py --algorithm graphsage --prefix ca-astroph --emb ${DATASPACE}/ca-astroph/unsup-graphsage/gcn_big_0.000010/ --tsv ${DATASPACE}/ca-astroph/tsv

# python data/export_tsv.py --algorithm node2vec --prefix ppi --emb ${DATASPACE}/ppi/emb/ --tsv ${DATASPACE}/ppi/tsv
# python data/export_tsv.py --algorithm graphsage --prefix ppi --emb ${DATASPACE}/ppi/unsup-graphsage/gcn_big_0.000010/ --tsv ${DATASPACE}/ppi/tsv

IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/erdos/prefix.txt))'
for i in "${PREFIX[@]}"
do
    python data/export_tsv.py --algorithm node2vec --prefix ${i} --emb ${DATASPACE}/erdos/emb/ --tsv ${DATASPACE}/erdos/tsv
done