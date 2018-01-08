LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
./darknet detector test cfg/coco.data cfg/yolo.cfg scripts/backup/yolo_100.weights data/person.jpg -thresh 0.9
