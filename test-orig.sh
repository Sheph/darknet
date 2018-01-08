LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
./darknet detector anomaly cfg/coco-orig.data cfg/yolo-orig.cfg yolo.weights /home/stas/Projects/1-pred/prednet/test3.avi -thresh 0.4
