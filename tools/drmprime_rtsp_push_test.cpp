#include "rtsp_mpp_sender.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

#include <libdrm/drm_fourcc.h>
#include "im2d.hpp"

namespace {

struct Options {
    std::string input_path;
    std::string rtsp_url = "rtsp://127.0.0.1:8554/drmprime-test";
    bool loop = false;
    bool pace = true;
    int max_frames = -1;
};

struct DecoderContext {
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

void PrintUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " <input.mp4> [rtsp-url] [--loop] [--no-pace] [--max-frames N]\n"
        << "Example:\n"
        << "  " << argv0
        << " /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/person.mp4"
        << " rtsp://127.0.0.1:8554/drmprime-test --loop\n";
}

bool ParseArgs(int argc, char** argv, Options* options) {
    if (!options || argc < 2) {
        return false;
    }

    options->input_path = argv[1];
    int idx = 2;
    if (idx < argc && std::strncmp(argv[idx], "--", 2) != 0) {
        options->rtsp_url = argv[idx++];
    }

    while (idx < argc) {
        std::string arg = argv[idx++];
        if (arg == "--loop") {
            options->loop = true;
        } else if (arg == "--no-pace") {
            options->pace = false;
        } else if (arg == "--max-frames" && idx < argc) {
            options->max_frames = std::stoi(argv[idx++]);
        } else {
            std::cerr << "[ERR] unknown arg: " << arg << "\n";
            return false;
        }
    }

    return true;
}

const char* PickDecoderName(AVCodecID codec_id) {
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

enum AVPixelFormat PreferDrmPrime(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    enum AVPixelFormat fallback = AV_PIX_FMT_NONE;
    bool saw_drm = false;
    bool saw_nv12 = false;

    std::cout << "[decode] get_format candidates:";
    for (const enum AVPixelFormat* p = pix_fmts; p && *p != AV_PIX_FMT_NONE; ++p) {
        const char* name = av_get_pix_fmt_name(*p);
        std::cout << " " << (name ? name : "unknown") << "(" << *p << ")";
        if (fallback == AV_PIX_FMT_NONE) fallback = *p;
        if (*p == AV_PIX_FMT_DRM_PRIME) saw_drm = true;
        if (*p == AV_PIX_FMT_NV12) saw_nv12 = true;
    }
    std::cout << "\n";

    if (saw_drm) {
        return AV_PIX_FMT_DRM_PRIME;
    }
    if (saw_nv12) {
        return AV_PIX_FMT_NV12;
    }
    return fallback;
}

void CloseDecoder(DecoderContext* ctx) {
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

bool OpenDecoder(const std::string& input_path, DecoderContext* ctx) {
    if (!ctx) return false;
    CloseDecoder(ctx);

    int ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_RKMPP, nullptr, nullptr, 0);
    std::cout << "[decode] av_hwdevice_ctx_create ret=" << ret << "\n";
    if (ret < 0) return false;

    ret = avformat_open_input(&ctx->format_ctx, input_path.c_str(), nullptr, nullptr);
    std::cout << "[decode] avformat_open_input ret=" << ret << " path=" << input_path << "\n";
    if (ret < 0) return false;

    ret = avformat_find_stream_info(ctx->format_ctx, nullptr);
    std::cout << "[decode] avformat_find_stream_info ret=" << ret << "\n";
    if (ret < 0) return false;

    ctx->video_stream = av_find_best_stream(ctx->format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    std::cout << "[decode] av_find_best_stream ret=" << ctx->video_stream << "\n";
    if (ctx->video_stream < 0) return false;

    AVStream* stream = ctx->format_ctx->streams[ctx->video_stream];
    const char* decoder_name = PickDecoderName(stream->codecpar->codec_id);
    if (!decoder_name) {
        std::cerr << "[ERR] unsupported codec id=" << stream->codecpar->codec_id << "\n";
        return false;
    }

    const AVCodec* codec = avcodec_find_decoder_by_name(decoder_name);
    std::cout << "[decode] decoder=" << decoder_name << " found=" << (codec ? 1 : 0) << "\n";
    if (!codec) return false;

    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) return false;

    ret = avcodec_parameters_to_context(ctx->codec_ctx, stream->codecpar);
    std::cout << "[decode] avcodec_parameters_to_context ret=" << ret << "\n";
    if (ret < 0) return false;

    ctx->codec_ctx->get_format = PreferDrmPrime;
    ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);

    ret = avcodec_open2(ctx->codec_ctx, codec, nullptr);
    std::cout << "[decode] avcodec_open2 ret=" << ret << "\n";
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

bool ConvertDrmPrimeFrameToBgr(const AVFrame* frame, cv::Mat* bgr_frame) {
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
        std::cerr << "[ERR] unsupported DRM format: 0x" << std::hex << drm_format << std::dec << "\n";
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
    if (!src_handle) {
        std::cerr << "[ERR] importbuffer_fd failed, fd=" << src_fd << "\n";
        return false;
    }
    if (!dst_handle) {
        std::cerr << "[ERR] importbuffer_virtualaddr failed, size=" << dst_size << "\n";
        releasebuffer_handle(src_handle);
        return false;
    }

    rga_buffer_t src_img = wrapbuffer_handle_t(src_handle, frame->width, frame->height,
                                               src_wstride, src_hstride, RK_FORMAT_YCbCr_420_SP);
    src_img.wstride = src_wstride;
    src_img.hstride = src_hstride;
    rga_buffer_t dst_img = wrapbuffer_handle_t(dst_handle, frame->width, frame->height,
                                               frame->width, frame->height, RK_FORMAT_BGR_888);

    IM_STATUS status = imcvtcolor(src_img, dst_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        std::cerr << "[ERR] imcvtcolor failed: " << imStrError(status) << "\n";
        return false;
    }

    return true;
}

bool ReceiveDecodedFrame(DecoderContext* ctx, cv::Mat* bgr_frame, int* drm_fd_out) {
    if (!ctx || !bgr_frame) return false;
    if (drm_fd_out) *drm_fd_out = -1;

    while (true) {
        int ret = avcodec_receive_frame(ctx->codec_ctx, ctx->frame);
        if (ret == AVERROR(EAGAIN)) {
            break;
        }
        if (ret == AVERROR_EOF) {
            return false;
        }
        if (ret < 0) {
            std::cerr << "[ERR] avcodec_receive_frame ret=" << ret << "\n";
            return false;
        }

        const char* fmt_name = av_get_pix_fmt_name((AVPixelFormat)ctx->frame->format);
        std::cout << "[decode] frame format=" << (fmt_name ? fmt_name : "unknown")
                  << " width=" << ctx->frame->width
                  << " height=" << ctx->frame->height << "\n";

        bool ok = false;
        if (ctx->frame->format == AV_PIX_FMT_DRM_PRIME) {
            AVDRMFrameDescriptor* desc = reinterpret_cast<AVDRMFrameDescriptor*>(ctx->frame->data[0]);
            if (desc && desc->nb_layers > 0 && desc->layers[0].nb_planes > 0) {
                int object_index = desc->layers[0].planes[0].object_index;
                int fd = (object_index >= 0 && object_index < (int)desc->nb_objects) ? desc->objects[object_index].fd : -1;
                uint32_t format = desc->layers[0].format;
                std::cout << "[decode] DRM layer fmt=0x" << std::hex << format << std::dec
                          << " fd=" << fd
                          << " pitch=" << desc->layers[0].planes[0].pitch << "\n";
                if (drm_fd_out) *drm_fd_out = fd;
            }
            ok = ConvertDrmPrimeFrameToBgr(ctx->frame, bgr_frame);
        } else {
            std::cerr << "[ERR] expected DRM_PRIME, got format=" << ctx->frame->format << "\n";
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
            std::cerr << "[ERR] avcodec_send_packet ret=" << ret << "\n";
            return false;
        }

        return ReceiveDecodedFrame(ctx, bgr_frame, drm_fd_out);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseArgs(argc, argv, &options)) {
        PrintUsage(argv[0]);
        return 1;
    }

    av_log_set_level(AV_LOG_ERROR);

    DecoderContext decoder;
    if (!OpenDecoder(options.input_path, &decoder)) {
        std::cerr << "[ERR] failed to open DRM_PRIME decoder for " << options.input_path << "\n";
        CloseDecoder(&decoder);
        return 1;
    }

    RtspMppSender sender;
    if (!sender.init(options.rtsp_url.c_str(), decoder.width, decoder.height, (int)(decoder.fps + 0.5))) {
        std::cerr << "[ERR] failed to init RTSP sender: " << options.rtsp_url << "\n";
        CloseDecoder(&decoder);
        return 1;
    }

    std::cout << "[push] input=" << options.input_path << "\n"
              << "[push] url=" << options.rtsp_url << "\n"
              << "[push] size=" << decoder.width << "x" << decoder.height
              << " fps=" << decoder.fps
              << " loop=" << (options.loop ? "true" : "false")
              << " pace=" << (options.pace ? "true" : "false") << "\n";

    const auto frame_interval = std::chrono::duration<double>(1.0 / std::max(1.0, decoder.fps));
    auto next_deadline = std::chrono::steady_clock::now();
    int pushed_frames = 0;

    while (true) {
        cv::Mat bgr_frame;
        int drm_fd = -1;
        if (!ReceiveDecodedFrame(&decoder, &bgr_frame, &drm_fd) || bgr_frame.empty()) {
            if (options.loop) {
                std::cout << "[push] loop restart\n";
                if (!OpenDecoder(options.input_path, &decoder)) {
                    std::cerr << "[ERR] failed to reopen decoder for loop\n";
                    break;
                }
                next_deadline = std::chrono::steady_clock::now();
                continue;
            }
            std::cout << "[push] decode finished\n";
            break;
        }

        if (!sender.push(bgr_frame)) {
            std::cerr << "[ERR] sender.push failed\n";
            break;
        }

        ++pushed_frames;
        if ((pushed_frames % 60) == 0) {
            std::cout << "[push] pushed_frames=" << pushed_frames
                      << " last_drm_fd=" << drm_fd << "\n";
        }

        if (options.max_frames > 0 && pushed_frames >= options.max_frames) {
            std::cout << "[push] reached max_frames=" << options.max_frames << "\n";
            break;
        }

        if (options.pace) {
            next_deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_interval);
            std::this_thread::sleep_until(next_deadline);
        }
    }

    sender.destroy();
    CloseDecoder(&decoder);
    return 0;
}
