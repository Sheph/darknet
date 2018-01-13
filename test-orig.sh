#LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
#./darknet detector anomaly cfg/coco-orig.data cfg/yolo-orig.cfg yolo.weights /home/stas/Projects/1-anomaly/Datasets/Pedestrian/test.avi -thresh 0.4

#LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
#./darknet detector anomaly cfg/coco-orig.data cfg/yolo-orig.cfg yolo.weights /home/stas/Projects/1-anomaly/z3.avi -thresh 0.25

LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
./darknet detector anomaly cfg/coco-orig.data cfg/yolo-orig.cfg yolo.weights /home/stas/Projects/1-anomaly/reception_long_train.avi -thresh 0.25
