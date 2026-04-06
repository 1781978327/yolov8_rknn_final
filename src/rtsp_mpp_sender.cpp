/**
 * RTSP + MPP 硬件推流，逻辑来自 rk_ffmpeg-main。将 BGR 帧经 MPP 编码后推送到 RTSP。
 * 每实例独立上下文，支持多路同时推流（双摄双 RTSP）。
 */
#include "rtsp_mpp_sender.h"
#include "command_mpp.h"
#include "ffmpeg_with_mpp.h"
#include <opencv2/imgproc.hpp>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_meta.h>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

using namespace cv;

typedef struct { MppPacket packet; AVBufferRef* encoder_ref; } RKMPPPacketContext;

struct MppRtspContext {
    unsigned int framecount = 0, width = 0, height = 0;
    int64_t last_pts = AV_NOPTS_VALUE;
    unsigned int hor_stride = 0, ver_stride = 0;
    unsigned int yuv_width = 0, yuv_height = 0, yuv_hor_stride = 0, yuv_ver_stride = 0;
    unsigned int image_size = 0;
    const AVCodec* codec = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFormatContext* formatCtx = nullptr;
    AVStream* stream = nullptr;
    AVBufferRef* hwdevice = nullptr;
    AVBufferRef* hwframe = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    long extra_data_size = 0;
    uint8_t* cExtradata = nullptr;
    AVPixelFormat hd_pix = AV_PIX_FMT_DRM_PRIME;
    AVPixelFormat sw_pix = AV_PIX_FMT_YUV420P;
    MppBufferGroup group = nullptr;
    MppBufferInfo info;
    MppBuffer buffer = nullptr;
    MppBuffer commitBuffer = nullptr;
    MppFrame mppframe = nullptr;
    MppPacket mppPacket = nullptr;
    MppApi* mppApi = nullptr;
    MppCtx mppCtx = nullptr;
    MppEncCfg cfg = nullptr;
    MppTask task = nullptr;
    MppMeta meta = nullptr;
    int mpp_initialized = 0;
    Command cmd;
};

char errInfo[200];
void print_error(int line, int res, const std::string& selfInfo) {
    av_strerror(res, errInfo, sizeof(errInfo));
    std::cout << "[ " << line << " ] " << errInfo << " " << selfInfo << std::endl;
}

static void rkmpp_release_packet(void* opaque, uint8_t* data) {
    (void)data;
    RKMPPPacketContext* p = (RKMPPPacketContext*)opaque;
    mpp_packet_deinit(&p->packet);
    if (p->encoder_ref) av_buffer_unref(&p->encoder_ref);
    av_free(p);
}

static int init_encoder(MppRtspContext* ctx, Command& obj) {
    int res = 0;
    unsigned int width = ctx->width, height = ctx->height;
    int fps = obj.get_fps() > 0 ? obj.get_fps() : 25;
    avformat_network_init();
    ctx->codec = avcodec_find_encoder_by_name("h264_rkmpp");
    if (!ctx->codec) ctx->codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!ctx->codec) { print_error(__LINE__, -1, "no H.264 encoder"); return -1; }
    ctx->codecCtx = avcodec_alloc_context3(ctx->codec);
    if (!ctx->codecCtx) { print_error(__LINE__, -1, "no codec context"); return -1; }
    ctx->codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    ctx->codecCtx->codec_id = AV_CODEC_ID_H264;
    ctx->codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
    ctx->codecCtx->width = width;
    ctx->codecCtx->height = height;
    ctx->codecCtx->bit_rate = 2 * 1024 * 1024;
    ctx->codecCtx->time_base = AVRational{ 1, fps };
    ctx->codecCtx->framerate = AVRational{ fps, 1 };
    ctx->codecCtx->gop_size = fps;
    ctx->codecCtx->max_b_frames = 0;
    ctx->codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->hwdevice = nullptr;
    ctx->hwframe = nullptr;
    if (strstr(ctx->codec->name, "rkmpp")) {
        res = av_hwdevice_ctx_create(&ctx->hwdevice, AV_HWDEVICE_TYPE_DRM, "/dev/dri/card0", nullptr, 0);
        if (res < 0) { print_error(__LINE__, res, "hw device"); return res; }
        ctx->codecCtx->pix_fmt = ctx->hd_pix;
        ctx->hwframe = av_hwframe_ctx_alloc(ctx->hwdevice);
        if (ctx->hwframe) {
            AVHWFramesContext* hwframeCtx = (AVHWFramesContext*)ctx->hwframe->data;
            hwframeCtx->format = ctx->hd_pix;
            hwframeCtx->sw_format = ctx->sw_pix;
            hwframeCtx->width = width;
            hwframeCtx->height = height;
            hwframeCtx->pool = av_buffer_pool_init(20 * sizeof(AVFrame), nullptr);
            res = av_hwframe_ctx_init(ctx->hwframe);
            if (res >= 0) {
                ctx->codecCtx->hw_frames_ctx = av_buffer_ref(ctx->hwframe);
                ctx->codecCtx->hw_device_ctx = av_buffer_ref(ctx->hwdevice);
                if (!ctx->codecCtx->hw_frames_ctx || !ctx->codecCtx->hw_device_ctx) {
                    if (ctx->codecCtx->hw_frames_ctx) av_buffer_unref(&ctx->codecCtx->hw_frames_ctx);
                    if (ctx->codecCtx->hw_device_ctx) av_buffer_unref(&ctx->codecCtx->hw_device_ctx);
                    print_error(__LINE__, AVERROR(ENOMEM), "hw context ref");
                    return AVERROR(ENOMEM);
                }
            }
        }
    }
    res = avformat_alloc_output_context2(&ctx->formatCtx, nullptr, "rtsp", obj.get_url());
    if (res < 0) { print_error(__LINE__, res, "output context"); return res; }
    ctx->stream = avformat_new_stream(ctx->formatCtx, ctx->codec);
    if (!ctx->stream) return -1;
    ctx->stream->time_base = AVRational{ 1, fps };
    ctx->stream->id = ctx->formatCtx->nb_streams - 1;
    res = avcodec_parameters_from_context(ctx->stream->codecpar, ctx->codecCtx);
    if (res < 0) return res;
    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "rtsp_transport", obj.get_trans_protocol(), 0);
    av_dict_set(&opt, "muxdelay", "0.1", 0);
    res = avformat_write_header(ctx->formatCtx, &opt);
    if (res < 0) { print_error(__LINE__, res, "write header"); return res; }
    av_dump_format(ctx->formatCtx, 0, obj.get_url(), 1);
    return res;
}

static MPP_RET init_mpp(MppRtspContext* ctx) {
    MPP_RET res = mpp_create(&ctx->mppCtx, &ctx->mppApi);
    int fps = ctx->cmd.get_fps() > 0 ? ctx->cmd.get_fps() : 25;
    if (res != MPP_OK) return res;
    res = mpp_init(ctx->mppCtx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (res != MPP_OK) return res;
    res = mpp_enc_cfg_init(&ctx->cfg);
    if (res != MPP_OK) return res;
    res = ctx->mppApi->control(ctx->mppCtx, MPP_ENC_GET_CFG, ctx->cfg);
    if (res != MPP_OK) return res;
    mpp_enc_cfg_set_s32(ctx->cfg, "prep:width", ctx->width);
    mpp_enc_cfg_set_s32(ctx->cfg, "prep:height", ctx->height);
    mpp_enc_cfg_set_s32(ctx->cfg, "prep:hor_stride", ctx->hor_stride);
    mpp_enc_cfg_set_s32(ctx->cfg, "prep:ver_stride", ctx->ver_stride);
    mpp_enc_cfg_set_s32(ctx->cfg, "prep:format", MPP_FMT_YUV420P);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:quality", MPP_ENC_RC_QUALITY_BEST);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:mode", MPP_ENC_RC_MODE_VBR);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_in_num", fps);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_in_denorm", 1);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_out_num", fps);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:fps_out_denorm", 1);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:bps_max", 10*1024*1024*17/16);
    mpp_enc_cfg_set_s32(ctx->cfg, "rc:bps_min", 10*1024*1024*15/16);
    mpp_enc_cfg_set_s32(ctx->cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(ctx->cfg, "h264:level", 42);
    ctx->mppApi->control(ctx->mppCtx, MPP_ENC_SET_CFG, ctx->cfg);
    MppPacket headpacket;
    RK_U8 enc_hdr_buf[1024];
    memset(enc_hdr_buf, 0, sizeof(enc_hdr_buf));
    mpp_packet_init(&headpacket, enc_hdr_buf, sizeof(enc_hdr_buf));
    res = ctx->mppApi->control(ctx->mppCtx, MPP_ENC_GET_HDR_SYNC, headpacket);
    void* ptr = mpp_packet_get_pos(headpacket);
    size_t len = mpp_packet_get_length(headpacket);
    ctx->extra_data_size = (long)len;
    ctx->cExtradata = (uint8_t*)malloc(ctx->extra_data_size);
    memcpy(ctx->cExtradata, ptr, len);
    mpp_packet_deinit(&headpacket);
    mpp_buffer_group_get_external(&ctx->group, MPP_BUFFER_TYPE_DRM);
    ctx->mpp_initialized = 1;
    std::cout << "init mpp encoder finished (YOLO RTSP)" << std::endl;
    return res;
}

static int init_data(MppRtspContext* ctx) { ctx->packet = av_packet_alloc(); return 0; }

static MPP_RET read_frame(MppRtspContext* ctx, cv::Mat& cvframe, void* ptr) {
    RK_U32 read_size = 0;
    RK_U8* buf_y = (RK_U8*)ptr;
    RK_U8* buf_u = buf_y + ctx->hor_stride * ctx->ver_stride;
    RK_U8* buf_v = buf_u + ctx->hor_stride * ctx->ver_stride / 4;
    for (RK_U32 row = 0; row < ctx->height; row++) {
        memcpy(buf_y + row * ctx->hor_stride, cvframe.datastart + read_size, ctx->width);
        read_size += ctx->width;
    }
    for (RK_U32 row = 0; row < ctx->height / 2; row++) {
        memcpy(buf_u + row * (ctx->hor_stride/2), cvframe.datastart + read_size, ctx->width/2);
        read_size += ctx->width/2;
    }
    for (RK_U32 row = 0; row < ctx->height / 2; row++) {
        memcpy(buf_v + row * (ctx->hor_stride/2), cvframe.datastart + read_size, ctx->width/2);
        read_size += ctx->width/2;
    }
    return MPP_OK;
}

static int send_packet(MppRtspContext* ctx, Command& obj) {
    (void)obj;
    // 按真实送帧时间生成时间戳，避免主循环抖动时仍然按固定 FPS 伪造时间轴，
    // 造成 VLC 先播空缓冲、再周期性等待下一批帧。
    int64_t now_us = av_gettime_relative();
    if (ctx->framecount == 0) {
        ctx->packet->pts = 0;
        ctx->packet->dts = 0;
        ctx->packet->duration = 1;
    } else {
        int64_t pts = av_rescale_q(now_us, AVRational{1, 1000000}, ctx->stream->time_base);
        if (ctx->last_pts != AV_NOPTS_VALUE && pts <= ctx->last_pts) {
            pts = ctx->last_pts + 1;
        }
        ctx->packet->duration = (ctx->last_pts == AV_NOPTS_VALUE) ? 1 : (pts - ctx->last_pts);
        ctx->packet->pts = pts;
        ctx->packet->dts = pts;
    }
    ctx->last_pts = ctx->packet->pts;

    ctx->framecount++;

    AVPacket out;
    av_init_packet(&out);

    // 默认直接使用编码出的 NAL
    int need_prepend = (ctx->cExtradata && ctx->extra_data_size > 0);
    int total_size = ctx->packet->size;
    if (need_prepend) {
        total_size += (int)ctx->extra_data_size;
    }

    int alloc_ret = av_new_packet(&out, total_size);
    if (alloc_ret < 0) {
        print_error(__LINE__, alloc_ret, "alloc out packet");
        return alloc_ret;
    }

    // av_new_packet 会重置时间戳字段，这里在分配后再写入，避免 RTSP 包出现无时间戳导致播放器卡顿。
    out.flags = ctx->packet->flags;
    out.pts = ctx->packet->pts;
    out.dts = ctx->packet->dts;
    out.duration = ctx->packet->duration;
    out.stream_index = ctx->stream ? ctx->stream->index : 0;
    out.pos = -1;

    uint8_t* dst = out.data;
    if (need_prepend) {
        memcpy(dst, ctx->cExtradata, ctx->extra_data_size);
        dst += ctx->extra_data_size;
    }
    memcpy(dst, ctx->packet->data, ctx->packet->size);

    int r = av_interleaved_write_frame(ctx->formatCtx, &out);
    if (r < 0) {
        print_error(__LINE__, r, "send packet");
    }
    av_packet_unref(&out);
    return r;
}

static MPP_RET convert_cvframe_to_drm(MppRtspContext* ctx, cv::Mat& cvframe, Command& obj) {
    MPP_RET res = mpp_buffer_get(nullptr, &ctx->buffer, ctx->image_size);
    if (res != MPP_OK) return res;
    ctx->info.fd = mpp_buffer_get_fd(ctx->buffer);
    ctx->info.ptr = mpp_buffer_get_ptr(ctx->buffer);
    ctx->info.index = ctx->framecount;
    ctx->info.size = ctx->image_size;
    ctx->info.type = MPP_BUFFER_TYPE_DRM;
    read_frame(ctx, cvframe, ctx->info.ptr);
    res = mpp_buffer_commit(ctx->group, &ctx->info);
    if (res != MPP_OK) return res;
    res = mpp_buffer_get(ctx->group, &ctx->commitBuffer, ctx->image_size);
    if (res != MPP_OK) return res;
    mpp_frame_init(&ctx->mppframe);
    mpp_frame_set_width(ctx->mppframe, ctx->width);
    mpp_frame_set_height(ctx->mppframe, ctx->height);
    mpp_frame_set_hor_stride(ctx->mppframe, ctx->yuv_hor_stride);
    mpp_frame_set_ver_stride(ctx->mppframe, ctx->ver_stride);
    mpp_frame_set_buf_size(ctx->mppframe, ctx->image_size);
    mpp_frame_set_buffer(ctx->mppframe, ctx->commitBuffer);
    mpp_frame_set_fmt(ctx->mppframe, MPP_FMT_YUV420P);
    mpp_frame_set_eos(ctx->mppframe, 0);
    mpp_packet_init_with_buffer(&ctx->mppPacket, ctx->commitBuffer);
    mpp_packet_set_length(ctx->mppPacket, 0);
    ctx->mppApi->poll(ctx->mppCtx, MPP_PORT_INPUT, MPP_POLL_BLOCK);
    ctx->mppApi->dequeue(ctx->mppCtx, MPP_PORT_INPUT, &ctx->task);
    mpp_task_meta_set_packet(ctx->task, KEY_OUTPUT_PACKET, ctx->mppPacket);
    mpp_task_meta_set_frame(ctx->task, KEY_INPUT_FRAME, ctx->mppframe);
    ctx->mppApi->enqueue(ctx->mppCtx, MPP_PORT_INPUT, ctx->task);
    ctx->mppApi->poll(ctx->mppCtx, MPP_PORT_OUTPUT, MPP_POLL_BLOCK);
    ctx->mppApi->dequeue(ctx->mppCtx, MPP_PORT_OUTPUT, &ctx->task);
    mpp_task_meta_get_packet(ctx->task, KEY_OUTPUT_PACKET, &ctx->mppPacket);
    ctx->mppApi->enqueue(ctx->mppCtx, MPP_PORT_OUTPUT, ctx->task);
    if (ctx->mppPacket) {
        RKMPPPacketContext* pkt_ctx = (RKMPPPacketContext*)av_mallocz(sizeof(*pkt_ctx));
        pkt_ctx->packet = ctx->mppPacket;
        pkt_ctx->encoder_ref = nullptr;
        ctx->packet->data = (uint8_t*)mpp_packet_get_data(ctx->mppPacket);
        ctx->packet->size = mpp_packet_get_length(ctx->mppPacket);
        ctx->packet->buf = av_buffer_create((uint8_t*)ctx->packet->data, ctx->packet->size, rkmpp_release_packet, pkt_ctx, AV_BUFFER_FLAG_READONLY);
        if (!ctx->packet->buf) {
            mpp_packet_deinit(&ctx->mppPacket);
            av_free(pkt_ctx);
            return MPP_NOK;
        }
        MppPacket pkt = pkt_ctx->packet;
        // packet 所有权已移交给 AVBuffer 的释放回调，避免后续误用/重复释放。
        ctx->mppPacket = nullptr;
        ctx->packet->pts = mpp_packet_get_pts(pkt);
        ctx->packet->dts = mpp_packet_get_dts(pkt);
        if (ctx->packet->pts <= 0) ctx->packet->pts = ctx->packet->dts;
        if (ctx->packet->dts <= 0) ctx->packet->dts = ctx->packet->pts;
        ctx->meta = mpp_packet_get_meta(pkt);
        int keyframe = 0;
        if (ctx->meta) mpp_meta_get_s32(ctx->meta, KEY_OUTPUT_INTRA, &keyframe);
        if (keyframe) ctx->packet->flags |= AV_PKT_FLAG_KEY;
    }
    int send_ret = send_packet(ctx, obj);
    if (send_ret < 0) {
        return MPP_NOK;
    }
    return res;
}

static int transfer_frame(MppRtspContext* ctx, cv::Mat& cvframe, Command& obj) {
    MPP_RET rc = convert_cvframe_to_drm(ctx, cvframe, obj);
    if (ctx->buffer) { mpp_buffer_put(ctx->buffer); ctx->buffer = nullptr; }
    if (ctx->commitBuffer) { mpp_buffer_put(ctx->commitBuffer); ctx->commitBuffer = nullptr; }
    mpp_buffer_group_clear(ctx->group);
    mpp_frame_deinit(&ctx->mppframe);
    return (rc == MPP_OK) ? 0 : -1;
}

static void context_destroy(MppRtspContext* ctx) {
    if (!ctx) return;
    if (ctx->mpp_initialized) {
        if (ctx->group) {
            mpp_buffer_group_put(ctx->group);
            ctx->group = nullptr;
        }
        ctx->mpp_initialized = 0;
    }
    if (ctx->cfg) {
        mpp_enc_cfg_deinit(ctx->cfg);
        ctx->cfg = nullptr;
    }
    if (ctx->mppCtx) {
        mpp_destroy(ctx->mppCtx);
        ctx->mppCtx = nullptr;
        ctx->mppApi = nullptr;
    }
    if (ctx->formatCtx) {
        av_write_trailer(ctx->formatCtx);
        if (ctx->formatCtx->pb) avio_closep(&ctx->formatCtx->pb);
        avformat_free_context(ctx->formatCtx);
        ctx->formatCtx = nullptr;
    }
    if (ctx->packet) { av_packet_free(&ctx->packet); ctx->packet = nullptr; }
    if (ctx->frame) { av_frame_free(&ctx->frame); ctx->frame = nullptr; }
    if (ctx->codecCtx) { avcodec_free_context(&ctx->codecCtx); ctx->codecCtx = nullptr; }
    if (ctx->cExtradata) { free(ctx->cExtradata); ctx->cExtradata = nullptr; }
    if (ctx->hwdevice) { av_buffer_unref(&ctx->hwdevice); ctx->hwdevice = nullptr; }
    if (ctx->hwframe) { av_buffer_unref(&ctx->hwframe); ctx->hwframe = nullptr; }
}

bool RtspMppSender::init(const char* rtsp_url, int w, int h, int fps) {
    if (inited_) return true;
    MppRtspContext* ctx = new MppRtspContext();
    ctx_ = ctx;
    ctx->width = w; ctx->height = h;
    ctx->yuv_width = w; ctx->yuv_height = h;
    ctx->hor_stride = MPP_ALIGN(ctx->width, 16);
    ctx->ver_stride = MPP_ALIGN(ctx->height, 16);
    ctx->yuv_hor_stride = ctx->hor_stride;
    ctx->yuv_ver_stride = ctx->ver_stride;
    ctx->image_size = (size_t)ctx->hor_stride * ctx->ver_stride * 3 / 2;
    ctx->framecount = 0;
    ctx->cmd.set_url(rtsp_url);
    ctx->cmd.set_fps(fps);
    ctx->cmd.set_width(ctx->width);
    ctx->cmd.set_height(ctx->height);
    ctx->cmd.set_use_hw(1);
    ctx->cmd.set_protocol("rtsp");
    ctx->cmd.set_trans_protocol("tcp");
    if (init_encoder(ctx, ctx->cmd) < 0) { context_destroy(ctx); delete ctx; ctx_ = nullptr; return false; }
    if (init_data(ctx) < 0) { context_destroy(ctx); delete ctx; ctx_ = nullptr; return false; }
    if (init_mpp(ctx) != MPP_OK) { context_destroy(ctx); delete ctx; ctx_ = nullptr; return false; }
    inited_ = true;
    return true;
}

bool RtspMppSender::push(cv::Mat& bgr_frame) {
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3) return false;
    if (!inited_ || !ctx_) return false;
    MppRtspContext* ctx = (MppRtspContext*)ctx_;
    if (bgr_frame.cols != (int)ctx->width || bgr_frame.rows != (int)ctx->height) {
        cv::Mat resized;
        cv::resize(bgr_frame, resized, cv::Size(ctx->width, ctx->height));
        bgr_frame = resized;
    }
    // 与 rk_ffmpeg-main 示例保持完全一致：直接用 COLOR_RGB2YUV_YV12，
    // OpenCV 默认输入是 BGR，这里故意保持同样的“错位”以得到一致的色彩效果。
    cv::Mat yuvframe;
    cv::cvtColor(bgr_frame, yuvframe, cv::COLOR_RGB2YUV_YV12);
    int ret = transfer_frame(ctx, yuvframe, ctx->cmd);
    if (ret != 0) {
        // 推流链路出现错误（如 Broken pipe）时，及时销毁上下文，
        // 让上层感知失败并触发重新初始化。
        destroy();
        return false;
    }
    av_packet_unref(ctx->packet);
    return true;
}

void RtspMppSender::destroy() {
    if (!inited_) return;
    if (ctx_) {
        context_destroy((MppRtspContext*)ctx_);
        delete (MppRtspContext*)ctx_;
        ctx_ = nullptr;
    }
    inited_ = false;
}
