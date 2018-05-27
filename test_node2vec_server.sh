# DATASPACE=/Users/tnguyen/dataspace/graph/
DATASPACE=/mnt/storage01/duong/dataspace/graph/
source activate python2

# python ./node2vec/src/main.py --input ${DATASPACE}/facebook/edgelist/facebook.edgelist --output ${DATASPACE}/facebook/emb/facebook.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/wikipedia/edgelist/POS.edgelist --output ${DATASPACE}/wikipedia/emb/POS.emb --weighted
# python ./node2vec/src/main.py --input ${DATASPACE}/blogcatalog/edgelist/blog.edgelist --output ${DATASPACE}/blogcatalog/emb/blog.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/ca-astroph/edgelist/ca-astroph.edgelist --output ${DATASPACE}/ca-astroph/emb/ca-astroph.emb
# python ./node2vec/src/main.py --input ${DATASPACE}/ppi/edgelist/ppi.edgelist --output ${DATASPACE}/ppi/emb/ppi.emb

# python ./node2vec/src/main.py --input ${DATASPACE}/reddit/edgelist/reddit-int.edgelist --output ${DATASPACE}/reddit/emb/reddit-int.emb
# python ./data/utils.py --emb ${DATASPACE}/reddit/emb/reddit-int.emb --id ${DATASPACE}/reddit/graphsage/reddit-id_map.json --out ${DATASPACE}/reddit/emb/reddit.emb

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/erdos/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python ./node2vec/src/main.py --input ${DATASPACE}/erdos/edgelist/${i}.edgelist --output ${DATASPACE}/erdos/emb/${i}.emb
# done

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/rr_graph/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python ./node2vec/src/main.py --input ${DATASPACE}/rr_graph/edgelist/${i}.edgelist --output ${DATASPACE}/rr_graph/emb/${i}.emb
# done

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/watts_graph/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python ./node2vec/src/main.py --input ${DATASPACE}/watts_graph/edgelist/${i}.edgelist --output ${DATASPACE}/watts_graph/emb/${i}.emb
# done

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/pa_graph/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python ./node2vec/src/main.py --input ${DATASPACE}/pa_graph/edgelist/${i}.edgelist --output ${DATASPACE}/pa_graph/emb/${i}.emb
# done

python ./node2vec/src/main.py --input ${DATASPACE}/pa_graph/edgelist/pa,n=10000,m=1.edgelist --output ${DATASPACE}/pa_graph/emb/pa,n=10000,m=1.emb

# python ./node2vec/src/main.py --input ${DATASPACE}/karate/edgelist/karate.edgelist --output ${DATASPACE}/karate/emb/karate.emb
# nice -n 19 
source activate base