DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow

# python data/convert_to_edgelist_format.py --reddit ${DATASPACE}/reddit --stat
# python data/convert_to_edgelist_format.py --blog ${DATASPACE}/blogcatalog --stat
# python data/convert_to_edgelist_format.py --ppi ${DATASPACE}/ppi --stat
# python data/convert_to_edgelist_format.py --wiki ${DATASPACE}/wikipedia --stat

source activate base

python data/convert_to_graphsage_format.py --facebook ${DATASPACE}/facebook