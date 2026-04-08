#include "http_ctrl_raw_video_rtsp.h"

#ifdef USE_RTSP_MPP

#include <algorithm>
#include <unistd.h>

#include <libdrm/drm_fourcc.h>

#include "im2d.h"
#include "im2d.hpp"
#include "rga.h"

namespace {

const char* PickRawVideoDecoderName(AVCodecID codec_id) {
    switch (codec_id) {
    case AV_CODEC_ID_H264:
        return "h264_rkmpp";
    case AV_CODEC_ID_HEVC:
        return "hevc_rkmpp";
    case AV_CODEC_ID_MJPEG:
        return "mjpeg_rkmpp";
    case AV_CODEC_ID_VP8:
        return "vp8_rkmpp";
    case AV_CODEC_ID_VP9:
        return "vp9_rkmpp";
    case AV_CODEC_ID_MPEG2VIDEO:
        return "mpeg2_rkmpp";
    case AV_CODEC_ID_MPEG4:
        return "mpeg4_rkmpp";
    default:
        return nullptr;
    }
}

enum AVPixelFormat PreferRawVideoDrmPrime(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    (void)ctx;
    enum AVPixelFormat fallback = AV_PIX_FMT_NONE;
    bool saw_drm = false;
    bool saw_nv12 = false;

    printf("[RTSP-RAW-VIDEO] get_format candidates:");
    for (const enum AVPixelFormat* p = pix_fmts; p && *p != AV_PIX_FMT_NONE; ++p) {
        const char* name = av_get_pix_fmt_name(*p);
        printf(" %s(%d)", name ? name : "unknown", *p);
        if (fallback == AV_PIX_FMT_NONE) fallback = *p;
        if (*p == AV_PIX_FMT_DRM_PRIME) saw_drm = true;
        if (*p == AV_PIX_FMT_NV12) saw_nv12 = true;
    }
    printf("\n");

    if (saw_drm) return AV_PIX_FMT_DRM_PRIME;
    if (saw_nv12) return AV_PIX_FMT_NV12;
    return fallback;
}

bool ConvertRawVideoDrmPrimeToBgr(const AVFrame* frame, cv::Mat* bgr_frame) {
    if (!frame || !bgr_frame || frame->format != AV_PIX_FMT_DRM_PRIME || !frame->data[0]) {
        return false;
    }

    AVDRMFrameDescriptor* desc = reinterpret_cast<AVDRMFrameDescriptor*>(frame->data[0]);
    if (!desc || desc->nb_layers < 1 || desc->layers[0].nb_planes < 2) {
        return false;
    }

    const int object_index = desc->layers[0].planes[0].object_index;
    if (object_index < 0 || object_index >= (int)desc->nb_objects) {
        return false;
    }

    const uint32_t drm_format = desc->layers[0].format;
    if (drm_format != DRM_FORMAT_NV12) {
        printf("[RTSP-RAW-VIDEO] unsupported DRM format: 0x%08x\n", drm_format);
        return false;
    }

    const int src_fd = desc->objects[object_index].fd;
    const int src_wstride = (int)desc->layers[0].planes[0].pitch;
    const int uv_offset = (int)desc->layers[0].planes[1].offset;
    int src_hstride = frame->height;
    if (src_wstride > 0 && uv_offset > 0) {
        src_hstride = std::max(frame->height, uv_offset / src_wstride);
    }

    if (src_fd < 0 || src_wstride <= 0) {
        return false;
    }

    bgr_frame->create(frame->height, frame->width, CV_8UC3);
    const int dst_size = bgr_frame->cols * bgr_frame->rows * 3;

    rga_buffer_handle_t src_handle = importbuffer_fd(src_fd, src_wstride, src_hstride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(bgr_frame->data, dst_size);
    if (!src_handle || !dst_handle) {
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        return false;
    }

    rga_buffer_t src_img = wrapbuffer_handle_t(src_handle, frame->width, frame->height,
                                               src_wstride, src_hstride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst_img = wrapbuffer_handle_t(dst_handle, frame->width, frame->height,
                                               frame->width, frame->height, RK_FORMAT_BGR_888);

    IM_STATUS status = imcvtcolor(src_img, dst_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        printf("[RTSP-RAW-VIDEO] imcvtcolor failed: %s\n", imStrError(status));
        return false;
    }
    return true;
}

bool ExtractRawVideoDrmPrimeInfo(const AVFrame* frame, RawVideoDmabufFrameInfo* frame_info) {
    if (!frame_info) return false;
    ClearRawVideoDmabufFrameInfo(frame_info);

    if (!frame || frame->format != AV_PIX_FMT_DRM_PRIME || !frame->data[0]) {
        return false;
    }

    AVDRMFrameDescriptor* desc = reinterpret_cast<AVDRMFrameDescriptor*>(frame->data[0]);
    if (!desc || desc->nb_layers < 1 || desc->layers[0].nb_planes < 2) {
        return false;
    }

    const int object_index = desc->layers[0].planes[0].object_index;
    if (object_index < 0 || object_index >= (int)desc->nb_objects) {
        return false;
    }

    const int src_fd = desc->objects[object_index].fd;
    const int dup_fd = dup(src_fd);
    if (dup_fd < 0) {
        return false;
    }

    const int src_wstride = (int)desc->layers[0].planes[0].pitch;
    const int uv_offset = (int)desc->layers[0].planes[1].offset;
    int src_hstride = frame->height;
    if (src_wstride > 0 && uv_offset > 0) {
        src_hstride = std::max(frame->height, uv_offset / src_wstride);
    }

    frame_info->valid = true;
    frame_info->fd = dup_fd;
    frame_info->size = (int)desc->objects[object_index].size;
    frame_info->width = frame->width;
    frame_info->height = frame->height;
    frame_info->wstride = src_wstride;
    frame_info->hstride = src_hstride;
    frame_info->drm_format = desc->layers[0].format;
    return true;
}

}  // namespace

void CloseRawVideoRtspDecoder(RawVideoRtspDecoderContext* ctx) {
    if (!ctx) return;
    av_packet_free(&ctx->packet);
    av_frame_free(&ctx->frame);
    avcodec_free_context(&ctx->codec_ctx);
    avformat_close_input(&ctx->format_ctx);
    av_buffer_unref(&ctx->hw_device_ctx);
    ctx->video_stream = -1;
    ctx->sent_eof = false;
    ctx->fps = 25.0;
    ctx->width = 0;
    ctx->height = 0;
}

bool OpenRawVideoRtspDecoder(const std::string& input_path, RawVideoRtspDecoderContext* ctx) {
    if (!ctx) return false;
    CloseRawVideoRtspDecoder(ctx);

    int ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_RKMPP, nullptr, nullptr, 0);
    printf("[RTSP-RAW-VIDEO] av_hwdevice_ctx_create ret=%d\n", ret);
    if (ret < 0) return false;

    ret = avformat_open_input(&ctx->format_ctx, input_path.c_str(), nullptr, nullptr);
    printf("[RTSP-RAW-VIDEO] avformat_open_input ret=%d path=%s\n", ret, input_path.c_str());
    if (ret < 0) return false;

    ret = avformat_find_stream_info(ctx->format_ctx, nullptr);
    if (ret < 0) return false;

    ctx->video_stream = av_find_best_stream(ctx->format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ctx->video_stream < 0) return false;

    AVStream* stream = ctx->format_ctx->streams[ctx->video_stream];
    const char* decoder_name = PickRawVideoDecoderName(stream->codecpar->codec_id);
    if (!decoder_name) return false;

    const AVCodec* codec = avcodec_find_decoder_by_name(decoder_name);
    if (!codec) return false;

    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) return false;

    ret = avcodec_parameters_to_context(ctx->codec_ctx, stream->codecpar);
    if (ret < 0) return false;

    ctx->codec_ctx->get_format = PreferRawVideoDrmPrime;
    ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);

    ret = avcodec_open2(ctx->codec_ctx, codec, nullptr);
    printf("[RTSP-RAW-VIDEO] avcodec_open2 ret=%d\n", ret);
    if (ret < 0) return false;

    ctx->packet = av_packet_alloc();
    ctx->frame = av_frame_alloc();
    if (!ctx->packet || !ctx->frame) return false;

    ctx->width = ctx->codec_ctx->width;
    ctx->height = ctx->codec_ctx->height;
    ctx->fps = av_q2d(stream->avg_frame_rate);
    if (ctx->fps <= 0.0) ctx->fps = av_q2d(stream->r_frame_rate);
    if (ctx->fps <= 0.0) ctx->fps = 25.0;
    ctx->sent_eof = false;
    return true;
}

void ClearRawVideoDmabufFrameInfo(RawVideoDmabufFrameInfo* frame_info) {
    if (!frame_info) return;
    if (frame_info->fd >= 0) {
        close(frame_info->fd);
    }
    *frame_info = RawVideoDmabufFrameInfo{};
}

bool ReadRawVideoRtspFrame(RawVideoRtspDecoderContext* ctx,
                           cv::Mat* bgr_frame,
                           RawVideoDmabufFrameInfo* frame_info) {
    if (!ctx) return false;
    if (bgr_frame) {
        bgr_frame->release();
    }
    if (frame_info) {
        ClearRawVideoDmabufFrameInfo(frame_info);
    }

    while (true) {
        int ret = avcodec_receive_frame(ctx->codec_ctx, ctx->frame);
        if (ret == AVERROR(EAGAIN)) {
            break;
        }
        if (ret == AVERROR_EOF) {
            return false;
        }
        if (ret < 0) {
            return false;
        }

        static int raw_video_decode_log_counter = 0;
        raw_video_decode_log_counter++;
        if (raw_video_decode_log_counter == 1 || (raw_video_decode_log_counter % 120) == 0) {
            const char* fmt_name = av_get_pix_fmt_name((AVPixelFormat)ctx->frame->format);
            printf("[RTSP-RAW-VIDEO] frame format=%s width=%d height=%d frame=%d\n",
                   fmt_name ? fmt_name : "unknown",
                   ctx->frame->width,
                   ctx->frame->height,
                   raw_video_decode_log_counter);
        }

        bool ok = false;
        if (ctx->frame->format == AV_PIX_FMT_DRM_PRIME) {
            bool dmabuf_ok = frame_info ? ExtractRawVideoDrmPrimeInfo(ctx->frame, frame_info) : false;
            bool bgr_ok = bgr_frame ? ConvertRawVideoDrmPrimeToBgr(ctx->frame, bgr_frame) : false;
            ok = dmabuf_ok || bgr_ok;
        }

        av_frame_unref(ctx->frame);
        return ok;
    }

    while (true) {
        if (ctx->sent_eof) {
            return false;
        }

        int ret = av_read_frame(ctx->format_ctx, ctx->packet);
        if (ret < 0) {
            avcodec_send_packet(ctx->codec_ctx, nullptr);
            ctx->sent_eof = true;
            continue;
        }

        if (ctx->packet->stream_index != ctx->video_stream) {
            av_packet_unref(ctx->packet);
            continue;
        }

        ret = avcodec_send_packet(ctx->codec_ctx, ctx->packet);
        av_packet_unref(ctx->packet);
        if (ret < 0) {
            return false;
        }

        return ReadRawVideoRtspFrame(ctx, bgr_frame, frame_info);
    }
}

#endif
