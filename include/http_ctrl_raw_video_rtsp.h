#ifndef HTTP_CTRL_RAW_VIDEO_RTSP_H
#define HTTP_CTRL_RAW_VIDEO_RTSP_H

#include <cstdint>
#include <string>

#include <opencv2/core/mat.hpp>

#ifdef USE_RTSP_MPP
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

struct RawVideoRtspDecoderContext {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    int video_stream = -1;
    bool sent_eof = false;
    double fps = 25.0;
    int width = 0;
    int height = 0;
};

struct RawVideoDmabufFrameInfo {
    bool valid = false;
    int fd = -1;
    int size = 0;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    uint32_t drm_format = 0;
};

void CloseRawVideoRtspDecoder(RawVideoRtspDecoderContext* ctx);
bool OpenRawVideoRtspDecoder(const std::string& input_path, RawVideoRtspDecoderContext* ctx);
void ClearRawVideoDmabufFrameInfo(RawVideoDmabufFrameInfo* frame_info);
bool ReadRawVideoRtspFrame(RawVideoRtspDecoderContext* ctx,
                           cv::Mat* bgr_frame,
                           RawVideoDmabufFrameInfo* frame_info = nullptr);
#endif

#endif
