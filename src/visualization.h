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

#ifdef OPENCV
void* open_video_stream(const char *f, int c, int w, int h, int fps);
void make_window(char *name, int w, int h, int fullscreen);
#endif

#endif // VISUALIZATION_H
