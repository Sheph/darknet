#LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
#./darknet detector test cfg/coco.data cfg/yolo.cfg scripts/backup/yolo_100.weights data/dog.jpg -thresh 0.9
./darknet detector demo cfg/coco.data cfg/yolo.cfg scripts/backup/yolo_100.weights /home/stas/Projects/1-pred/prednet/test3.avi -thresh 0.9
