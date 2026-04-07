#ifndef HTTP_CTRL_CAMERA_IO_H
#define HTTP_CTRL_CAMERA_IO_H

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include "v4l2_dmabuf_capture.h"

struct CameraDmabufFrameInfo {
    bool valid = false;
    int index = -1;
    int fd = -1;
    void* va = nullptr;
    int size = 0;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    int rga_format = 0;
};

void release_camera_dmabuf_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                                 CameraDmabufFrameInfo* frame_info);

bool acquire_camera_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                          cv::VideoCapture* cv_cap,
                          cv::Mat* frame_out,
                          CameraDmabufFrameInfo* frame_info = nullptr);

bool read_camera_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                       cv::VideoCapture* cv_cap,
                       cv::Mat* frame_out);

#endif
