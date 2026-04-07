#include "v4l2_dmabuf_capture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#else
#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif
#endif

namespace v4l2_dmabuf {
namespace {

int dma_buf_alloc(const std::string& heap_path, size_t size, int* fd_out, void** va_out) {
    int heap_fd = open(heap_path.c_str(), O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        return -1;
    }

    struct dma_heap_allocation_data alloc_data;
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        close(heap_fd);
        return -1;
    }
    close(heap_fd);

    void* va = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc_data.fd, 0);
    if (va == MAP_FAILED) {
        close(alloc_data.fd);
        return -1;
    }

    *fd_out = alloc_data.fd;
    *va_out = va;
    return 0;
}

void dma_buf_free(int fd, void* va, size_t size) {
    if (va && va != MAP_FAILED) {
        munmap(va, size);
    }
    if (fd >= 0) {
        close(fd);
    }
}

std::string fourcc_to_string(uint32_t v) {
    std::string s(4, ' ');
    s[0] = static_cast<char>(v & 0xFF);
    s[1] = static_cast<char>((v >> 8) & 0xFF);
    s[2] = static_cast<char>((v >> 16) & 0xFF);
    s[3] = static_cast<char>((v >> 24) & 0xFF);
    return s;
}

}  // namespace

void close_capture(CaptureContext* cap) {
    if (!cap) return;
    if (cap->fd >= 0 && cap->streaming) {
        int type = cap->buf_type;
        ioctl(cap->fd, VIDIOC_STREAMOFF, &type);
    }
    cap->streaming = false;
    if (cap->fd >= 0) {
        close(cap->fd);
        cap->fd = -1;
    }
    for (auto& b : cap->buffers) {
        dma_buf_free(b.fd, b.va, b.size);
        b.fd = -1;
        b.va = nullptr;
        b.size = 0;
    }
    cap->buffers.clear();
}

int queue_buffer(CaptureContext* cap, int index) {
    if (!cap || cap->fd < 0 || index < 0 || index >= (int)cap->buffers.size()) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = cap->buf_type;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = static_cast<uint32_t>(index);

    if (cap->mplane) {
        struct v4l2_plane planes[VIDEO_MAX_PLANES];
        memset(planes, 0, sizeof(planes));
        buf.length = static_cast<uint32_t>(cap->num_planes);
        buf.m.planes = planes;
        planes[0].m.fd = cap->buffers[index].fd;
        planes[0].length = static_cast<uint32_t>(cap->plane0_size);
        planes[0].bytesused = static_cast<uint32_t>(cap->plane0_size);
    } else {
        buf.length = static_cast<uint32_t>(cap->plane0_size);
        buf.m.fd = cap->buffers[index].fd;
    }

    return ioctl(cap->fd, VIDIOC_QBUF, &buf);
}

int open_capture(const CaptureConfig& config, CaptureContext* out, std::string* err) {
    if (!out) return -1;
    *out = CaptureContext{};

    int fd = open(config.device_name.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        if (err) *err = "open failed";
        return -1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        if (err) *err = "VIDIOC_QUERYCAP failed";
        close(fd);
        return -1;
    }

    uint32_t caps = cap.capabilities;
    if (caps & V4L2_CAP_DEVICE_CAPS) caps = cap.device_caps;
    if (!(caps & V4L2_CAP_STREAMING)) {
        if (err) *err = "device does not support streaming";
        close(fd);
        return -1;
    }

    int buf_type = -1;
    bool mplane = false;
    if (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        mplane = true;
    } else if (caps & V4L2_CAP_VIDEO_CAPTURE) {
        buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        mplane = false;
    } else {
        if (err) *err = "device has no VIDEO_CAPTURE capability";
        close(fd);
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = buf_type;
    if (mplane) {
        fmt.fmt.pix_mp.width = static_cast<uint32_t>(config.width);
        fmt.fmt.pix_mp.height = static_cast<uint32_t>(config.height);
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
    } else {
        fmt.fmt.pix.width = static_cast<uint32_t>(config.width);
        fmt.fmt.pix.height = static_cast<uint32_t>(config.height);
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
    }
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) != 0) {
        if (err) *err = "VIDIOC_S_FMT failed";
        close(fd);
        return -1;
    }

    const uint32_t actual_pixfmt = mplane ? fmt.fmt.pix_mp.pixelformat : fmt.fmt.pix.pixelformat;
    if (actual_pixfmt != V4L2_PIX_FMT_YUYV) {
        if (err) *err = "camera returned unsupported pixfmt=" + fourcc_to_string(actual_pixfmt);
        close(fd);
        return -1;
    }

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = buf_type;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(std::max(1, config.fps));
    (void)ioctl(fd, VIDIOC_S_PARM, &parm);

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = static_cast<uint32_t>(std::max(2, config.dma_buffers));
    req.type = buf_type;
    req.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) {
        if (err) *err = "VIDIOC_REQBUFS failed";
        close(fd);
        return -1;
    }

    const uint32_t fmt_width = mplane ? fmt.fmt.pix_mp.width : fmt.fmt.pix.width;
    const uint32_t fmt_height = mplane ? fmt.fmt.pix_mp.height : fmt.fmt.pix.height;
    const uint32_t fmt_sizeimage = mplane ? fmt.fmt.pix_mp.plane_fmt[0].sizeimage : fmt.fmt.pix.sizeimage;
    const uint32_t fmt_bytesperline = mplane ? fmt.fmt.pix_mp.plane_fmt[0].bytesperline : fmt.fmt.pix.bytesperline;
    const size_t plane0_size = static_cast<size_t>(
        std::max<uint32_t>(fmt_sizeimage, static_cast<uint32_t>(fmt_width * fmt_height * 2)));

    out->fd = fd;
    out->buf_type = buf_type;
    out->mplane = mplane;
    out->width = static_cast<int>(fmt_width);
    out->height = static_cast<int>(fmt_height);
    out->bytesperline = static_cast<int>(fmt_bytesperline > 0 ? fmt_bytesperline : fmt_width * 2);
    out->wstride = out->bytesperline / 2;
    out->hstride = out->height;
    out->pixfmt = actual_pixfmt;
    out->num_planes = mplane ? static_cast<int>(fmt.fmt.pix_mp.num_planes) : 1;
    out->plane0_size = plane0_size;
    out->buffers.resize(req.count);

    for (uint32_t i = 0; i < req.count; ++i) {
        auto& b = out->buffers[i];
        b.size = plane0_size;
        if (dma_buf_alloc(config.dma_heap, plane0_size, &b.fd, &b.va) != 0) {
            if (err) *err = "dma_buf_alloc failed";
            close_capture(out);
            return -1;
        }
        if (queue_buffer(out, static_cast<int>(i)) != 0) {
            if (err) *err = "VIDIOC_QBUF failed";
            close_capture(out);
            return -1;
        }
    }

    int type = buf_type;
    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
        if (err) *err = "VIDIOC_STREAMON failed";
        close_capture(out);
        return -1;
    }

    out->streaming = true;
    return 0;
}

int dequeue_buffer(CaptureContext* cap, int timeout_ms, int* index_out) {
    if (!cap || cap->fd < 0 || !index_out) return -1;
    *index_out = -1;

    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = cap->fd;
    pfd.events = POLLIN | POLLERR;
    int pr = poll(&pfd, 1, timeout_ms);
    if (pr == 0) return 0;
    if (pr < 0) {
        if (errno == EINTR) return 0;
        return -1;
    }
    if (pfd.revents & POLLERR) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = cap->buf_type;
    buf.memory = V4L2_MEMORY_DMABUF;

    struct v4l2_plane planes[VIDEO_MAX_PLANES];
    if (cap->mplane) {
        memset(planes, 0, sizeof(planes));
        buf.length = static_cast<uint32_t>(cap->num_planes);
        buf.m.planes = planes;
    }

    if (ioctl(cap->fd, VIDIOC_DQBUF, &buf) != 0) {
        if (errno == EAGAIN || errno == EINTR) return 0;
        return -1;
    }
    if (buf.index >= cap->buffers.size()) return -1;
    *index_out = static_cast<int>(buf.index);
    return 1;
}

}  // namespace v4l2_dmabuf
