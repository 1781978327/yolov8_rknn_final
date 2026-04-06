#include "opencv2/opencv.hpp"
#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"

class PreResize
{
public:
    PreResize();
    PreResize(int, int, int);
    void init(double, double);
    int input_height;
    int input_width;
    int input_channel;
    double fx;
    double fy;
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    void resize(cv::Mat &img, cv::Mat &_img);
};
