#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network *net;
static image buff [3];
static image buff_letter[3];
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_done = 0;
static float *avg;
static double demo_time;

static FILE* json_file = NULL;
static float target_fps = 2.0f;
static int first = 1;

static void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[0].data;
    float *prediction = network_predict(net, X);
    l.output = prediction;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net->w, net->h, demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[0];
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes);

    int i,j;
    int cur_frame = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES);
    int fdet = 1;

    if (first)
        fprintf(json_file, "[%d, [\n", cur_frame);
    else
        fprintf(json_file, ",\n[%d, [\n", cur_frame);

    for (i = 0; i < demo_detections; ++i) {
        char labelstr[4096] = {0};
        int class = -1;
        float this_prob = 0.0f;
        for (j = 0; j < demo_classes; ++j) {
            if (probs[i][j] > demo_thresh) {
                if (class < 0) {
                    strcat(labelstr, demo_names[j]);
                    class = j;
                    this_prob = probs[i][j];
                }
            }
        }
        if (class >= 0) {
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*display.w;
            int right = (b.x+b.w/2.)*display.w;
            int top   = (b.y-b.h/2.)*display.h;
            int bot   = (b.y+b.h/2.)*display.h;

            if(left < 0) left = 0;
            if(right > display.w-1) right = display.w-1;
            if(top < 0) top = 0;
            if(bot > display.h-1) bot = display.h-1;

            if (!fdet) {
                fprintf(json_file, ",\n");
            }

            fprintf(json_file, "[\"%s\", %d, %f, %f, %f, %f, %f]", labelstr, class, this_prob, left / (float)display.w, top / (float)display.h, (right - left) / (float)display.w, (bot - top) / (float)display.h);
            fdet = 0;
        }
    }

    fprintf(json_file, "]]");

    first = 0;
    running = 0;
    return 0;
}

static void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[0]);
    letterbox_image_into(buff[0], net->w, net->h, buff_letter[0]);
    if(status == 0) demo_done = 1;
    return 0;
}

static void *display_in_thread(void *ptr)
{
    show_image_cv(buff[0], "Anomaly", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void anomaly(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Anomaly\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    json_file = fopen("boxes.json", "wb");

    fprintf(json_file, "[\n");

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    float actual_fps = 25;

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Anomaly", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Anomaly", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Anomaly", 0, 0);
        }
    }

    demo_time = what_time_is_it_now();

    j = 0;

    while (!demo_done) {
        ++j;
        if (j < (int)(actual_fps / target_fps)) {
            IplImage* src1 = cvQueryFrame(cap);
            if (!src1) {
                demo_done = 1;
            }
            continue;
        }
        fetch_in_thread(NULL);
        j = 0;

        detect_in_thread(NULL);
        fps = 1./(what_time_is_it_now() - demo_time);
        demo_time = what_time_is_it_now();
        display_in_thread(0);
        ++count;
    }

    printf("Done!\n");

    fprintf(json_file, "]\n");
    fclose(json_file);
}
