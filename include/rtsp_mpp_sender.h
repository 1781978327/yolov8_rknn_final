#ifndef RTSP_MPP_SENDER_H
#define RTSP_MPP_SENDER_H

#include <cstdint>
#include <opencv2/core.hpp>

/** 使用 MPP 硬件编码 + FFmpeg 推送到 RTSP，支持 BGR Mat / BGR dma-buf / NV12 dma-buf 输入。 */
class RtspMppSender {
public:
    RtspMppSender() = default;
    ~RtspMppSender() { destroy(); }
    bool init(const char* rtsp_url, int width, int height, int fps = 30);
    bool push(cv::Mat& bgr_frame);
    bool push_bgr_dmabuf(int fd, int width, int height, int wstride, int hstride);
    bool push_dmabuf(int fd, int size, int width, int height, int wstride, int hstride, uint32_t drm_format);
    void destroy();
    bool inited() const { return inited_; }
private:
    bool inited_ = false;
    void* ctx_ = nullptr;  // 每实例独立上下文，支持多路同时推流
};

#endif
