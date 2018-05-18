# DATASPACE=/Users/tnguyen/dataspace/graph
DATASPACE=/mnt/storage01/duong/dataspace/graph
source activate tensorflow
cd graphsage/
python -m graphsage.utils ${DATASPACE}/wikipedia/graphsage/POS-G.json ${DATASPACE}/wikipedia/POS-walks.txt
python -m graphsage.utils ${DATASPACE}/reddit/graphsage/reddit-G.json ${DATASPACE}/reddit/reddit-walks.txt
python -m graphsage.utils ${DATASPACE}/ppi/graphsage/ppi-G.json ${DATASPACE}/ppi/ppi-walks.txt
cd ../
source activate base