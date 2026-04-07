#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>

#include <libdrm/drm_fourcc.h>

static const char* pick_decoder_name(enum AVCodecID codec_id) {
    switch (codec_id) {
    case AV_CODEC_ID_H264:
        return "h264_rkmpp";
    case AV_CODEC_ID_HEVC:
        return "hevc_rkmpp";
    case AV_CODEC_ID_MPEG4:
        return "mpeg4_rkmpp";
    default:
        return NULL;
    }
}

static enum AVPixelFormat prefer_drm_prime(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p = pix_fmts;
    enum AVPixelFormat fallback = AV_PIX_FMT_NONE;
    int saw_drm = 0;
    int saw_nv12 = 0;

    printf("[probe] get_format candidates:");
    for (; p && *p != AV_PIX_FMT_NONE; ++p) {
        const char* name = av_get_pix_fmt_name(*p);
        printf(" %s(%d)", name ? name : "unknown", *p);
        if (fallback == AV_PIX_FMT_NONE) fallback = *p;
        if (*p == AV_PIX_FMT_DRM_PRIME) saw_drm = 1;
        if (*p == AV_PIX_FMT_NV12) saw_nv12 = 1;
    }
    printf("\n");

    if (saw_drm) {
        if (ctx->hw_device_ctx) {
            AVBufferRef* frames_ref = NULL;
            int ret = avcodec_get_hw_frames_parameters(ctx, ctx->hw_device_ctx,
                                                       AV_PIX_FMT_DRM_PRIME, &frames_ref);
            printf("[probe] avcodec_get_hw_frames_parameters(drm_prime) ret=%d\n", ret);
            if (ret >= 0 && frames_ref) {
                AVHWFramesContext* frames_ctx = (AVHWFramesContext*)frames_ref->data;
                if (frames_ctx) {
                    frames_ctx->initial_pool_size = 8;
                }
                ret = av_hwframe_ctx_init(frames_ref);
                printf("[probe] av_hwframe_ctx_init ret=%d\n", ret);
                if (ret >= 0) {
                    av_buffer_unref(&ctx->hw_frames_ctx);
                    ctx->hw_frames_ctx = av_buffer_ref(frames_ref);
                    printf("[probe] attached hw_frames_ctx for drm_prime\n");
                }
                av_buffer_unref(&frames_ref);
            }
        }
        return AV_PIX_FMT_DRM_PRIME;
    }

    if (saw_nv12) return AV_PIX_FMT_NV12;
    return fallback;
}

int main(int argc, char** argv) {
    const char* input_path =
        argc > 1 ? argv[1]
                 : "/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/person.mp4";

    AVFormatContext* fmt = NULL;
    AVCodecContext* dec_ctx = NULL;
    AVBufferRef* hw_device_ctx = NULL;
    AVPacket* pkt = NULL;
    AVFrame* frame = NULL;
    int video_stream = -1;
    int ret = 0;

    av_log_set_level(AV_LOG_ERROR);

    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_RKMPP, NULL, NULL, 0);
    printf("[probe] av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_RKMPP) ret=%d\n", ret);
    if (ret < 0) goto end;

    ret = avformat_open_input(&fmt, input_path, NULL, NULL);
    printf("[probe] avformat_open_input ret=%d path=%s\n", ret, input_path);
    if (ret < 0) goto end;

    ret = avformat_find_stream_info(fmt, NULL);
    printf("[probe] avformat_find_stream_info ret=%d\n", ret);
    if (ret < 0) goto end;

    video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    printf("[probe] av_find_best_stream ret=%d\n", video_stream);
    if (video_stream < 0) {
        ret = video_stream;
        goto end;
    }

    const char* decoder_name = pick_decoder_name(fmt->streams[video_stream]->codecpar->codec_id);
    if (!decoder_name) {
        printf("[probe] unsupported codec id=%d\n", fmt->streams[video_stream]->codecpar->codec_id);
        ret = -1;
        goto end;
    }

    const AVCodec* decoder = avcodec_find_decoder_by_name(decoder_name);
    printf("[probe] decoder=%s found=%d\n", decoder_name, decoder ? 1 : 0);
    if (!decoder) {
        ret = -1;
        goto end;
    }

    dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    ret = avcodec_parameters_to_context(dec_ctx, fmt->streams[video_stream]->codecpar);
    printf("[probe] avcodec_parameters_to_context ret=%d\n", ret);
    if (ret < 0) goto end;

    dec_ctx->get_format = prefer_drm_prime;
    dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    printf("[probe] hw_device_ctx attached=%d\n", dec_ctx->hw_device_ctx ? 1 : 0);

    ret = avcodec_open2(dec_ctx, decoder, NULL);
    printf("[probe] avcodec_open2 ret=%d\n", ret);
    if (ret < 0) goto end;

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    while ((ret = av_read_frame(fmt, pkt)) >= 0) {
        if (pkt->stream_index != video_stream) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(dec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0) {
            printf("[probe] avcodec_send_packet ret=%d\n", ret);
            break;
        }

        while ((ret = avcodec_receive_frame(dec_ctx, frame)) >= 0) {
            const char* name = av_get_pix_fmt_name((enum AVPixelFormat)frame->format);
            printf("[probe] receive frame format=%s(%d) width=%d height=%d\n",
                   name ? name : "unknown", frame->format, frame->width, frame->height);

            if (frame->format == AV_PIX_FMT_DRM_PRIME && frame->data[0]) {
                AVDRMFrameDescriptor* desc = (AVDRMFrameDescriptor*)frame->data[0];
                printf("[probe] DRM_PRIME desc=%p nb_objects=%d nb_layers=%d\n",
                       (void*)desc, desc ? (int)desc->nb_objects : -1,
                       desc ? (int)desc->nb_layers : -1);
                if (desc && desc->nb_layers > 0 && desc->layers[0].nb_planes > 0) {
                    int obj_idx = desc->layers[0].planes[0].object_index;
                    uint32_t format = desc->layers[0].format;
                    printf("[probe] DRM layer fmt=0x%08x obj_idx=%d fd=%d pitch=%ld\n",
                           format,
                           obj_idx,
                           (obj_idx >= 0 && obj_idx < (int)desc->nb_objects) ? desc->objects[obj_idx].fd : -1,
                           (long)desc->layers[0].planes[0].pitch);
                    if (format == DRM_FORMAT_NV12) {
                        printf("[probe] DRM layer format check: NV12 OK\n");
                    }
                }
            } else {
                printf("[probe] non-DRM frame linesize[0]=%d linesize[1]=%d\n",
                       frame->linesize[0], frame->linesize[1]);
            }

            av_frame_unref(frame);
            ret = 0;
            goto end;
        }

        if (ret == AVERROR(EAGAIN)) {
            continue;
        }
        if (ret == AVERROR_EOF) {
            ret = 0;
            break;
        }
        if (ret < 0) {
            printf("[probe] avcodec_receive_frame ret=%d\n", ret);
            break;
        }
    }

end:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt);
    av_buffer_unref(&hw_device_ctx);
    return ret < 0 ? 1 : 0;
}
