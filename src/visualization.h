#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "image.h"
#include "convolutional_layer.h"

image **load_alphabet();
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void draw_label(image a, int r, int c, image label, const float *rgb);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);

void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
void show_image_normalized(image im, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void visualize_network(network *net);

#ifndef __cplusplus
#ifdef OPENCV
image get_image_from_stream(CvCapture *cap);
int fill_image_from_stream(CvCapture *cap, image im);
void flush_stream_buffer(CvCapture *cap, int n);
void show_image_cv(image p, const char *name, IplImage *disp);
#endif
#endif

#endif // VISUALIZATION_H
