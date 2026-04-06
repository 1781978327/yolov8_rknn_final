#ifndef RTSP_MPP_SENDER_H
#define RTSP_MPP_SENDER_H

#include <opencv2/core.hpp>

/** 使用 MPP 硬件编码 + FFmpeg 将 BGR 帧推送到 RTSP。用法：init(url,w,h,fps) -> push(bgr_mat) -> destroy() */
class RtspMppSender {
public:
    RtspMppSender() = default;
    ~RtspMppSender() { destroy(); }
    bool init(const char* rtsp_url, int width, int height, int fps = 30);
    bool push(cv::Mat& bgr_frame);
    void destroy();
    bool inited() const { return inited_; }
private:
    bool inited_ = false;
    void* ctx_ = nullptr;  // 每实例独立上下文，支持多路同时推流
};

#endif
