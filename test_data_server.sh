DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow

python data/convert_to_edgelist_format.py --ppi ${DATASPACE}/ppi --stat
python data/convert_to_edgelist_format.py --wikipedia ${DATASPACE}/wikipedia --stat
python data/convert_to_edgelist_format.py --reddit ${DATASPACE}/reddit --stat

source activate base