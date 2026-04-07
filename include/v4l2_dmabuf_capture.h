#ifndef V4L2_DMABUF_CAPTURE_H
#define V4L2_DMABUF_CAPTURE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace v4l2_dmabuf {

struct DmaBuffer {
    int fd = -1;
    void* va = nullptr;
    size_t size = 0;
};

struct CaptureConfig {
    int width = 640;
    int height = 480;
    int fps = 30;
    int dma_buffers = 4;
    std::string device_name = "/dev/video0";
    std::string dma_heap = "/dev/dma_heap/system";
};

struct CaptureContext {
    int fd = -1;
    bool streaming = false;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    int bytesperline = 0;
    uint32_t pixfmt = 0;
    int buf_type = 0;
    bool mplane = false;
    int num_planes = 0;
    size_t plane0_size = 0;
    std::vector<DmaBuffer> buffers;
};

int open_capture(const CaptureConfig& config, CaptureContext* out, std::string* err = nullptr);
void close_capture(CaptureContext* cap);
int queue_buffer(CaptureContext* cap, int index);
int dequeue_buffer(CaptureContext* cap, int timeout_ms, int* index_out);

}  // namespace v4l2_dmabuf

#endif  // V4L2_DMABUF_CAPTURE_H
