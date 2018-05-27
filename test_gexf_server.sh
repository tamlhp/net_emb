DATASPACE=/mnt/storage01/duong/dataspace/graph

python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/facebook/edgelist/facebook.edgelist --gexf ${DATASPACE}/facebook/gexf/facebook.gexf
python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/wikipedia/edgelist/POS.edgelist --gexf ${DATASPACE}/wikipedia/gexf/POS.gexf --weighted
python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/blogcatalog/edgelist/blog.edgelist --gexf ${DATASPACE}/blogcatalog/gexf/blog.gexf
python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/ca-astroph/edgelist/ca-astroph.edgelist --gexf ${DATASPACE}/ca-astroph/gexf/ca-astroph.gexf
python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/ppi/edgelist/ppi.edgelist --gexf ${DATASPACE}/ppi/gexf/ppi.gexf
python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/reddit/edgelist/reddit.edgelist --gexf ${DATASPACE}/reddit/gexf/reddit.gexf

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/watts_graph/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/watts_graph/edgelist/${i}.edgelist --gexf ${DATASPACE}/watts_graph/gexf/${i}.gexf
# done

# IFS=$'\r\n' GLOBIGNORE='*' command eval  'PREFIX=($(cat ${DATASPACE}/pa_graph/prefix.txt))'
# for i in "${PREFIX[@]}"
# do
#     python data/convert_to_gexf_format.py --edgelist ${DATASPACE}/pa_graph/edgelist/${i}.edgelist --gexf ${DATASPACE}/pa_graph/gexf/${i}.gexf
# done