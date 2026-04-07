#include "http_ctrl_camera_io.h"

#include <opencv2/imgproc.hpp>

#include "rga.h"

void release_camera_dmabuf_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                                 CameraDmabufFrameInfo* frame_info) {
    if (!frame_info) return;
    if (dmabuf_cap && dmabuf_cap->fd >= 0 && dmabuf_cap->streaming &&
        frame_info->valid && frame_info->index >= 0) {
        (void)v4l2_dmabuf::queue_buffer(dmabuf_cap, frame_info->index);
    }
    *frame_info = CameraDmabufFrameInfo{};
}

bool acquire_camera_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                          cv::VideoCapture* cv_cap,
                          cv::Mat* frame_out,
                          CameraDmabufFrameInfo* frame_info) {
    if (!frame_out) return false;
    frame_out->release();
    if (frame_info) {
        *frame_info = CameraDmabufFrameInfo{};
    }

    if (dmabuf_cap && dmabuf_cap->fd >= 0 && dmabuf_cap->streaming) {
        int index = -1;
        int ret = v4l2_dmabuf::dequeue_buffer(dmabuf_cap, 100, &index);
        if (ret == 1 && index >= 0 && index < (int)dmabuf_cap->buffers.size()) {
            auto& buffer = dmabuf_cap->buffers[index];
            if (frame_info) {
                frame_info->valid = true;
                frame_info->index = index;
                frame_info->fd = buffer.fd;
                frame_info->va = buffer.va;
                frame_info->size = (int)buffer.size;
                frame_info->width = dmabuf_cap->width;
                frame_info->height = dmabuf_cap->height;
                frame_info->wstride = dmabuf_cap->wstride > 0 ? dmabuf_cap->wstride : dmabuf_cap->width;
                frame_info->hstride = dmabuf_cap->hstride > 0 ? dmabuf_cap->hstride : dmabuf_cap->height;
                frame_info->rga_format = RK_FORMAT_YUYV_422;
            }
            if (buffer.va && dmabuf_cap->width > 0 && dmabuf_cap->height > 0) {
                size_t step = (size_t)(dmabuf_cap->bytesperline > 0 ? dmabuf_cap->bytesperline : dmabuf_cap->width * 2);
                cv::Mat yuyv(dmabuf_cap->height, dmabuf_cap->width, CV_8UC2, buffer.va, step);
                cv::cvtColor(yuyv, *frame_out, cv::COLOR_YUV2BGR_YUYV);
            }
            return !frame_out->empty();
        }
        return false;
    }

    if (cv_cap && cv_cap->isOpened()) {
        return cv_cap->read(*frame_out) && !frame_out->empty();
    }

    return false;
}

bool read_camera_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                       cv::VideoCapture* cv_cap,
                       cv::Mat* frame_out) {
    CameraDmabufFrameInfo frame_info;
    bool ok = acquire_camera_frame(dmabuf_cap, cv_cap, frame_out, &frame_info);
    release_camera_dmabuf_frame(dmabuf_cap, &frame_info);
    return ok;
}
