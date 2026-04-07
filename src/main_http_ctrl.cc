// HTTP/WebSocket 控制服务 - 多线程推理版本
// 编译: cmake . && make

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <glob.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <memory>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <libdrm/drm_fourcc.h>

// RKNN 相关
#include "rknnPool.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "im2d.hpp"
#include "rga.h"
#include "v4l2_dmabuf_capture.h"

// RTSP 推流相关
#ifdef USE_RTSP_MPP
#include "rtsp_mpp_sender.h"
#include "ffmpeg_rkmpp_reader.h"
#endif

#ifdef USE_RTSP_MPP
namespace {

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

static const char* PickRawVideoDecoderName(AVCodecID codec_id) {
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

static enum AVPixelFormat PreferRawVideoDrmPrime(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
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

static void CloseRawVideoRtspDecoder(RawVideoRtspDecoderContext* ctx) {
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

static bool OpenRawVideoRtspDecoder(const std::string& input_path, RawVideoRtspDecoderContext* ctx) {
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

static bool ConvertRawVideoDrmPrimeToBgr(const AVFrame* frame, cv::Mat* bgr_frame) {
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

static void ClearRawVideoDmabufFrameInfo(RawVideoDmabufFrameInfo* frame_info) {
    if (!frame_info) return;
    if (frame_info->fd >= 0) {
        close(frame_info->fd);
    }
    *frame_info = RawVideoDmabufFrameInfo{};
}

static bool ExtractRawVideoDrmPrimeInfo(const AVFrame* frame, RawVideoDmabufFrameInfo* frame_info) {
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

static bool ReadRawVideoRtspFrame(RawVideoRtspDecoderContext* ctx, cv::Mat* bgr_frame,
                                  RawVideoDmabufFrameInfo* frame_info = nullptr) {
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
            if (dmabuf_ok) {
                ok = true;
            } else if (bgr_frame) {
                ok = ConvertRawVideoDrmPrimeToBgr(ctx->frame, bgr_frame);
            }
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

}  // namespace
#endif

// ---------------------- 配置 ----------------------
#define DEFAULT_HTTP_PORT  8091
#define DEFAULT_RTSP_HOST "127.0.0.1"
#define DEFAULT_RTSP_PORT 8554
#define DEFAULT_MODEL_PATH "../model/RK3588/yolov8s.rknn"
#define DEFAULT_LABEL_PATH "/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/coco_80_labels_list.txt"
#define DEFAULT_WIDTH     640
#define DEFAULT_HEIGHT    640
#define SLOTS_PER_CAM     3  // 每摄像头 3 个 slot

int g_http_port = DEFAULT_HTTP_PORT;
int g_server_fd = -1;
std::string g_rtsp_host = DEFAULT_RTSP_HOST;
int g_rtsp_port = DEFAULT_RTSP_PORT;
std::string g_rtsp_url_0 = "rtsp://127.0.0.1:8554/cam0";
std::string g_rtsp_url_1 = "rtsp://127.0.0.1:8554/cam1";
std::string g_rtsp_url_video = "rtsp://127.0.0.1:8554/cam3";

// ---------------------- 全局状态 ----------------------
std::atomic<bool> g_running(true);
std::atomic<bool> g_inference_enabled(false);  // 默认关闭推理
std::atomic<bool> g_model_loaded(false);
std::atomic<bool> g_model_loading(false);
std::atomic<int>  g_current_cam(0);
std::atomic<int>  g_cam0_fps(0), g_cam1_fps(0);
std::mutex g_model_mutex;
std::string g_model_path = DEFAULT_MODEL_PATH;
std::string g_label_path = "";
std::string g_model_error;
std::vector<rknn_lite*> g_rkpool;
std::atomic<int> g_active_jobs(0);
std::atomic<bool> g_model_switching(false);

// 最新带检测框的帧（供 HTTP API 获取）
cv::Mat g_latest_frame;
std::mutex g_latest_frame_mutex;
std::atomic<bool> g_frame_available(false);
std::atomic<int> g_latest_frame_slot(-1);  // g_latest_frame 对应的 slot，下游绘制时避免错配

// 跟踪器相关
std::atomic<bool> g_tracker_enabled(false);  // 0=关闭, 1=开启
std::mutex g_tracker_mutex;
std::vector<std::vector<TrackerResultItem>> g_tracker_results;  // 每个 slot 的跟踪结果

// RTSP 推流
#ifdef USE_RTSP_MPP
std::atomic<bool> g_rtsp_streaming(false);
RtspMppSender* g_rtsp_sender0 = nullptr;
RtspMppSender* g_rtsp_sender1 = nullptr;
RtspMppSender* g_rtsp_sender_video = nullptr;  // 视频模式专用推流
std::thread g_video_rtsp_raw_thread;
std::atomic<bool> g_video_rtsp_raw_running(false);
std::atomic<bool> g_video_rtsp_raw_stop(false);
std::mutex g_rtsp_mutex;

extern std::atomic<bool> g_video_mode;
extern std::atomic<bool> g_video_loop;
extern std::string g_video_path;

void destroy_rtsp_sender(RtspMppSender*& sender) {
    if (!sender) return;
    sender->destroy();
    delete sender;
    sender = nullptr;
}

void stop_rtsp_senders_locked() {
    destroy_rtsp_sender(g_rtsp_sender0);
    destroy_rtsp_sender(g_rtsp_sender1);
    destroy_rtsp_sender(g_rtsp_sender_video);
}

void stop_video_rtsp_raw_thread() {
    g_video_rtsp_raw_stop = true;
    if (g_video_rtsp_raw_thread.joinable()) {
        g_video_rtsp_raw_thread.join();
    }
    g_video_rtsp_raw_running = false;
}

void video_rtsp_raw_loop() {
    g_video_rtsp_raw_running = true;
    g_video_rtsp_raw_stop = false;

    RawVideoRtspDecoderContext decoder;
    if (!OpenRawVideoRtspDecoder(g_video_path, &decoder)) {
        printf("[RTSP-RAW-VIDEO] 无法打开视频: %s\n", g_video_path.c_str());
        g_video_rtsp_raw_running = false;
        return;
    }

    double fps = decoder.fps;
    if (fps <= 0.0) fps = 25.0;
    auto frame_interval = std::chrono::microseconds((int)(1000000.0 / fps));
    auto next_deadline = std::chrono::steady_clock::now();
    int pushed = 0;

    while (!g_video_rtsp_raw_stop.load() && g_running.load()) {
        if (!g_rtsp_streaming.load() || !g_video_mode.load() || g_model_loaded.load()) {
            break;
        }

        cv::Mat frame;
        RawVideoDmabufFrameInfo frame_info;
        if (!ReadRawVideoRtspFrame(&decoder, &frame, &frame_info) ||
            (!frame_info.valid && frame.empty())) {
            ClearRawVideoDmabufFrameInfo(&frame_info);
            if (g_video_loop.load()) {
                CloseRawVideoRtspDecoder(&decoder);
                if (!OpenRawVideoRtspDecoder(g_video_path, &decoder)) {
                    printf("[RTSP-RAW-VIDEO] 循环重开失败: %s\n", g_video_path.c_str());
                    break;
                }
                fps = decoder.fps;
                if (fps <= 0.0) fps = 25.0;
                frame_interval = std::chrono::microseconds((int)(1000000.0 / fps));
                next_deadline = std::chrono::steady_clock::now();
                continue;
            }
            printf("[RTSP-RAW-VIDEO] 视频播放完毕\n");
            break;
        }

        bool pushed_ok = false;
        {
            std::lock_guard<std::mutex> lock(g_rtsp_mutex);
            if (g_rtsp_sender_video && g_rtsp_sender_video->inited()) {
                if (frame_info.valid) {
                    pushed_ok = g_rtsp_sender_video->push_dmabuf(
                        frame_info.fd,
                        frame_info.size,
                        frame_info.width,
                        frame_info.height,
                        frame_info.wstride,
                        frame_info.hstride,
                        frame_info.drm_format);
                } else if (!frame.empty()) {
                    pushed_ok = g_rtsp_sender_video->push(frame);
                }
            }
        }
        ClearRawVideoDmabufFrameInfo(&frame_info);
        if (!pushed_ok) {
            printf("[RTSP-RAW-VIDEO] 推流写包失败，停止裸流线程\n");
            break;
        }

        ++pushed;
        if ((pushed % 120) == 0) {
            printf("[RTSP-RAW-VIDEO] pushed=%d fps=%.1f\n", pushed, fps);
        }

        next_deadline += frame_interval;
        std::this_thread::sleep_until(next_deadline);
    }

    CloseRawVideoRtspDecoder(&decoder);
    g_video_rtsp_raw_running = false;
}

void start_video_rtsp_raw_thread_if_needed() {
    if (g_video_rtsp_raw_running.load()) {
        return;
    }
    g_video_rtsp_raw_thread = std::thread(video_rtsp_raw_loop);
}
#endif

// 视频文件处理
struct VideoFrameInfo {
    bool valid = false;
    int fd = -1;
    bool nv12_valid = false;
    cv::Mat nv12_packed;
    int size = 0;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    uint32_t drm_format = 0;
};

std::atomic<bool> g_video_mode(false);  // 是否为视频文件模式
std::atomic<bool> g_video_loop(false);   // 是否循环播放视频
std::string g_video_path;                 // 当前视频路径
cv::VideoCapture g_video_cap;            // 视频捕获对象
#ifdef USE_RTSP_MPP
std::unique_ptr<FFmpegRkmppReader> g_video_hw_reader;
std::atomic<bool> g_video_hw_enabled(false);
#endif
std::mutex g_video_mutex;
std::queue<cv::Mat> g_video_frames;      // 视频帧队列
std::condition_variable g_video_cv;
std::thread g_video_thread;             // 视频读取线程
std::atomic<bool> g_video_running(false);
std::atomic<int> g_video_width(1920);
std::atomic<int> g_video_height(1080);
std::atomic<int> g_video_fps(25);

// 检测阈值设置
std::atomic<float> g_confidence_threshold(0.5f);  // 置信度阈值
std::atomic<int> g_cam0_detection_count(0);  // 摄像头0检测数量
std::atomic<int> g_cam1_detection_count(0);  // 摄像头1检测数量

// 当前帧的图像尺寸（用于坐标转换）
cv::Size g_current_img_size(1280, 720);  // 默认值
int g_model_width = 640, g_model_height = 640;  // 模型输入尺寸（YOLOv8 默认 640）

// ---------------------- 工具函数 ----------------------
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *localtime(&tt);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

long get_process_rss_kb() {
    std::ifstream ifs("/proc/self/status");
    if (!ifs.is_open()) return -1;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long rss_kb = -1;
            iss >> rss_kb;
            return rss_kb;
        }
    }
    return -1;
}

void refresh_rtsp_urls() {
    g_rtsp_url_0 = "rtsp://" + g_rtsp_host + ":" + std::to_string(g_rtsp_port) + "/cam0";
    g_rtsp_url_1 = "rtsp://" + g_rtsp_host + ":" + std::to_string(g_rtsp_port) + "/cam1";
    g_rtsp_url_video = "rtsp://" + g_rtsp_host + ":" + std::to_string(g_rtsp_port) + "/cam3";
}

std::string url_decode(const std::string& str) {
    std::string result;
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '%' && i + 2 < str.size()) {
            int val;
            std::istringstream iss(str.substr(i + 1, 2));
            iss >> std::hex >> val;
            result += char(val);
            i += 2;
        } else if (str[i] == '+') {
            result += ' ';
        } else {
            result += str[i];
        }
    }
    return result;
}

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

bool file_exists(const std::string& path) {
    return !path.empty() && access(path.c_str(), R_OK) == 0;
}

struct CameraDmabufFrameInfo {
    bool valid = false;
    int index = -1;
    int fd = -1;
    void* va = nullptr;
    int size = 0;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    int rga_format = 0;
};

static void release_camera_dmabuf_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                                        CameraDmabufFrameInfo* frame_info) {
    if (!frame_info) return;
    if (dmabuf_cap && dmabuf_cap->fd >= 0 && dmabuf_cap->streaming &&
        frame_info->valid && frame_info->index >= 0) {
        (void)v4l2_dmabuf::queue_buffer(dmabuf_cap, frame_info->index);
    }
    *frame_info = CameraDmabufFrameInfo{};
}

static bool acquire_camera_frame(v4l2_dmabuf::CaptureContext* dmabuf_cap,
                                 cv::VideoCapture* cv_cap,
                                 cv::Mat* frame_out,
                                 CameraDmabufFrameInfo* frame_info = nullptr) {
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

std::string parse_query_param(const std::string& path, const std::string& key) {
    size_t qpos = path.find('?');
    if (qpos == std::string::npos) return "";
    std::string query = path.substr(qpos + 1);
    std::string needle = key + "=";
    size_t pos = query.find(needle);
    if (pos == std::string::npos) return "";
    size_t value_start = pos + needle.size();
    size_t value_end = query.find('&', value_start);
    std::string raw = query.substr(value_start, value_end == std::string::npos ? std::string::npos : (value_end - value_start));
    return url_decode(raw);
}

std::string detect_model_path_locked() {
    // 若外部已指定模型路径，优先使用该路径
    if (file_exists(g_model_path)) {
        return g_model_path;
    }

    std::vector<std::string> candidates = {
        DEFAULT_MODEL_PATH,
        "../model/RK3588/yolov8n.rknn",
        "../model/RK3588/yolov8m.rknn"
    };

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    if (glob("../model/RK3588/*.rknn", 0, nullptr, &glob_result) == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            candidates.emplace_back(glob_result.gl_pathv[i]);
        }
    }
    globfree(&glob_result);

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    // 让 yolov8s 优先被选中
    std::stable_sort(candidates.begin(), candidates.end(), [](const std::string& a, const std::string& b) {
        bool a_s = a.find("yolov8s") != std::string::npos;
        bool b_s = b.find("yolov8s") != std::string::npos;
        if (a_s != b_s) return a_s;
        return a < b;
    });

    for (const auto& path : candidates) {
        if (file_exists(path)) return path;
    }
    return "";
}

void release_model_workers_locked() {
    for (auto*& ptr : g_rkpool) {
        delete ptr;
        ptr = nullptr;
    }
}

bool ensure_model_loaded(std::string* msg_out = nullptr) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (g_model_loaded.load()) {
        if (msg_out) *msg_out = "模型已就绪";
        return true;
    }
    if (g_model_loading.load()) {
        if (msg_out) *msg_out = "模型正在加载中";
        return false;
    }

    g_model_loading = true;
    g_model_error.clear();

    std::string model_to_use = detect_model_path_locked();
    if (model_to_use.empty()) {
        g_model_loading = false;
        g_model_loaded = false;
        g_model_error = "未找到可用模型(.rknn)，请检查 ../model/RK3588/";
        if (msg_out) *msg_out = g_model_error;
        return false;
    }

    printf("[Model] 开始加载模型: %s\n", model_to_use.c_str());
    if (g_rkpool.empty()) {
        g_rkpool.resize(SLOTS_PER_CAM * 2, nullptr);
    }

    for (size_t i = 0; i < g_rkpool.size(); ++i) {
        if (!g_rkpool[i]) {
            int core_id = (int)(i % 3);
            g_rkpool[i] = new rknn_lite(const_cast<char*>(model_to_use.c_str()), core_id);
            printf("[Model] Worker %zu 初始化完成 (核心 %d)\n", i, core_id);
        }
    }

    g_model_path = model_to_use;
    g_model_loaded = true;
    g_model_loading = false;
    if (msg_out) *msg_out = "模型加载成功";
    printf("[Model] 加载完成: %s\n", g_model_path.c_str());
    return true;
}

bool unload_model_runtime(std::string* msg_out = nullptr) {
    g_inference_enabled = false;
    g_tracker_enabled = false;
    g_model_switching = true;

    // 给主循环一个时间片停止继续派发任务
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (g_active_jobs.load() > 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (g_active_jobs.load() > 0) {
        g_model_switching = false;
        if (msg_out) *msg_out = "模型卸载超时：仍有推理任务在运行";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        release_model_workers_locked();
        g_model_loaded = false;
        g_model_loading = false;
        g_model_error.clear();
    }
    rknn_lite::reset_shared_deepsort_trackers();
    rknn_lite::reset_shared_bytetrack_trackers();
    {
        std::lock_guard<std::mutex> lock(g_tracker_mutex);
        for (auto& tracks : g_tracker_results) {
            tracks.clear();
        }
    }
    g_cam0_detection_count = 0;
    g_cam1_detection_count = 0;
    g_model_switching = false;

    if (msg_out) *msg_out = "模型已卸载";
    printf("[Model] 已卸载\n");
    return true;
}

std::string build_json_response(const std::string& status, const std::string& message = "") {
    std::ostringstream oss;
    oss << "{\"status\":\"" << status << "\"";
    if (!message.empty()) {
        oss << ",\"message\":\"" << message << "\"";
    }
    oss << ",\"timestamp\":\"" << get_timestamp() << "\"";
    oss << "}";
    return oss.str();
}

// 解析简单的 JSON body
bool parse_json_body(const std::string& body, std::string& path, bool& loop) {
    // 查找 "path": "xxx"
    size_t path_pos = body.find("\"path\"");
    if (path_pos != std::string::npos) {
        size_t colon = body.find(":", path_pos);
        if (colon != std::string::npos) {
            size_t quote1 = body.find("\"", colon + 1);
            if (quote1 != std::string::npos) {
                size_t quote2 = body.find("\"", quote1 + 1);
                if (quote2 != std::string::npos) {
                    path = body.substr(quote1 + 1, quote2 - quote1 - 1);
                }
            }
        }
    }

    // 查找 "loop": true/false
    size_t loop_pos = body.find("\"loop\"");
    if (loop_pos != std::string::npos) {
        size_t colon = body.find(":", loop_pos);
        if (colon != std::string::npos) {
            size_t value_start = colon + 1;
            while (value_start < body.size() && isspace(body[value_start])) value_start++;
            loop = (body.substr(value_start, 4) == "true");
        }
    }

    return !path.empty();
}

std::string build_status_response() {
    std::ostringstream oss;
    oss << "{";
    oss << "\"running\":" << (g_running ? "true" : "false") << ",";
    oss << "\"inference_enabled\":" << (g_inference_enabled ? "true" : "false") << ",";
    oss << "\"current_camera\":" << g_current_cam.load() << ",";
    oss << "\"fps\":" << (g_cam0_fps.load() + g_cam1_fps.load()) << ",";
    oss << "\"fps_cam0\":" << g_cam0_fps.load() << ",";
        oss << "\"fps_cam1\":" << g_cam1_fps.load();
#ifdef USE_RTSP_MPP
    oss << ",\"rtsp_streaming\":" << (g_rtsp_streaming ? "true" : "false");
#else
    oss << ",\"rtsp_streaming\":false";
#endif
    oss << "}";
    return oss.str();
}

void clear_video_queue_locked() {
    while (!g_video_frames.empty()) {
        g_video_frames.pop();
    }
}

bool open_video_source_locked() {
#ifdef USE_RTSP_MPP
    if (!g_video_hw_reader) {
        g_video_hw_reader.reset(new FFmpegRkmppReader());
    }
    if (g_video_hw_reader->Open(g_video_path)) {
        if (g_video_hw_reader->Width() > 0) g_video_width = g_video_hw_reader->Width();
        if (g_video_hw_reader->Height() > 0) g_video_height = g_video_hw_reader->Height();
        if (g_video_hw_reader->Fps() >= 1.0 && g_video_hw_reader->Fps() <= 120.0) {
            g_video_fps = (int)(g_video_hw_reader->Fps() + 0.5);
        }
        g_video_hw_enabled = true;
        printf("[Video] FFmpeg DRM_PRIME 已打开: %dx%d @ %.1f fps (decoder=%s)\n",
               g_video_width.load(), g_video_height.load(), g_video_hw_reader->Fps(),
               g_video_hw_reader->DecoderName().c_str());
        return true;
    }
    if (g_video_hw_reader) {
        g_video_hw_reader->Close();
    }
    g_video_hw_enabled = false;
#endif

    g_video_cap.open(g_video_path, cv::CAP_FFMPEG);
    if (!g_video_cap.isOpened()) {
        return false;
    }

    int vw = (int)g_video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int vh = (int)g_video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double vf = g_video_cap.get(cv::CAP_PROP_FPS);
    if (vw > 0) g_video_width = vw;
    if (vh > 0) g_video_height = vh;
    if (vf >= 1.0 && vf <= 120.0) g_video_fps = (int)(vf + 0.5);
    printf("[Video] OpenCV/FFmpeg 回退已打开: %dx%d @ %.1f fps\n", g_video_width.load(), g_video_height.load(), vf);
    return true;
}

bool read_video_frame_locked(cv::Mat& frame, VideoFrameInfo* frame_info = nullptr) {
    if (frame_info) {
        *frame_info = VideoFrameInfo{};
    }
#ifdef USE_RTSP_MPP
    if (g_video_hw_enabled.load() && g_video_hw_reader) {
        FFmpegRkmppReader::FrameInfo reader_info;
        FFmpegRkmppReader::FrameInfo* reader_info_ptr = frame_info ? &reader_info : nullptr;
        auto copy_reader_info = [&]() {
            if (!frame_info || !reader_info_ptr) return;
            frame_info->nv12_valid = reader_info_ptr->nv12_valid;
            frame_info->nv12_packed = reader_info_ptr->nv12_packed;
            frame_info->valid = reader_info_ptr->valid;
            frame_info->fd = reader_info_ptr->fd;
            frame_info->size = reader_info_ptr->size;
            frame_info->width = reader_info_ptr->width;
            frame_info->height = reader_info_ptr->height;
            frame_info->wstride = reader_info_ptr->wstride;
            frame_info->hstride = reader_info_ptr->hstride;
            frame_info->drm_format = reader_info_ptr->drm_format;
        };

        if (g_video_hw_reader->ReadFrame(frame, reader_info_ptr) && !frame.empty()) {
            copy_reader_info();
            return true;
        }
        if (g_video_loop.load()) {
            g_video_hw_reader->Close();
            if (g_video_hw_reader->Open(g_video_path) && g_video_hw_reader->ReadFrame(frame, reader_info_ptr) && !frame.empty()) {
                if (g_video_hw_reader->Width() > 0) g_video_width = g_video_hw_reader->Width();
                if (g_video_hw_reader->Height() > 0) g_video_height = g_video_hw_reader->Height();
                if (g_video_hw_reader->Fps() >= 1.0 && g_video_hw_reader->Fps() <= 120.0) {
                    g_video_fps = (int)(g_video_hw_reader->Fps() + 0.5);
                }
                copy_reader_info();
                return true;
            }
        }
        g_video_hw_reader->Close();
        g_video_hw_enabled = false;
        return false;
    }
#endif

    if (!g_video_cap.isOpened()) {
        return false;
    }
    if (g_video_cap.read(frame) && !frame.empty()) {
        return true;
    }
    if (g_video_loop.load()) {
        g_video_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        return g_video_cap.read(frame) && !frame.empty();
    }
    return false;
}

bool ensure_video_source_open_locked() {
    bool video_opened =
#ifdef USE_RTSP_MPP
        (g_video_hw_enabled.load() && g_video_hw_reader && g_video_hw_reader->IsOpen()) ||
#endif
        g_video_cap.isOpened();
    if (video_opened) {
        return true;
    }
    return open_video_source_locked();
}

void close_video_source_locked() {
#ifdef USE_RTSP_MPP
    if (g_video_hw_reader) {
        g_video_hw_reader->Close();
    }
    g_video_hw_enabled = false;
#endif
    if (g_video_cap.isOpened()) {
        g_video_cap.release();
    }
}

void video_reader_loop() {
    const size_t kMaxQueuedFrames = 8;

    while (g_running.load()) {
        cv::Mat frame;
        bool frame_ok = false;

        {
            std::unique_lock<std::mutex> lock(g_video_mutex);
            if (!g_video_running.load()) {
                break;
            }

            bool video_opened =
#ifdef USE_RTSP_MPP
                (g_video_hw_enabled.load() && g_video_hw_reader && g_video_hw_reader->IsOpen()) ||
#endif
                g_video_cap.isOpened();
            if (!video_opened) {
                if (!ensure_video_source_open_locked()) {
                    printf("[Video] 无法打开视频: %s\n", g_video_path.c_str());
                    g_video_running = false;
                    g_video_mode = false;
                    clear_video_queue_locked();
                    break;
                }
            }

            frame_ok = read_video_frame_locked(frame);

            if (!frame_ok) {
                close_video_source_locked();
                g_video_running = false;
                g_video_mode = false;
                clear_video_queue_locked();
                printf("[Video] 视频播放完毕\n");
                break;
            }

            if (g_video_frames.size() >= kMaxQueuedFrames) {
                g_video_frames.pop();
            }
            g_video_frames.push(frame.clone());
        }

        g_video_cv.notify_one();
    }

    g_video_cv.notify_all();
}

void stop_video_reader() {
#ifdef USE_RTSP_MPP
    stop_video_rtsp_raw_thread();
#endif
    {
        std::lock_guard<std::mutex> lock(g_video_mutex);
        g_video_running = false;
        g_video_mode = false;
        clear_video_queue_locked();
        close_video_source_locked();
    }
    g_video_cv.notify_all();
    if (g_video_thread.joinable()) {
        g_video_thread.join();
    }
}

void start_video_reader(const std::string& path, bool loop) {
    stop_video_reader();
    {
        std::lock_guard<std::mutex> lock(g_video_mutex);
        g_video_path = path;
        g_video_loop = loop;
        g_video_mode = true;
        g_video_running = true;
        g_video_width = 1920;
        g_video_height = 1080;
        g_video_fps = 25;
        clear_video_queue_locked();
    }
}

// 绘制带跟踪的检测框（需要外部持有 g_tracker_mutex）
void draw_tracker_boxes(cv::Mat& img, const std::vector<TrackerResultItem>& tracks, int slot_idx) {
    static int debug_counter = 0;
    bool print_debug = (debug_counter++ % 300 == 0);
    if (print_debug) {
        printf("[DEBUG draw] 收到 %zu 个跟踪结果, 图像尺寸: %dx%d, slot=%d\n",
               tracks.size(), img.cols, img.rows, slot_idx);
    }

    char text[256];
    int max_boxes = 20;
    int box_count = 0;
    for (const auto& track : tracks) {
        if (!track.active || track.track_id < 0) continue;

        box_count++;
        if (box_count > max_boxes) break;

        int x1 = std::max(0, std::min((int)track.x1, img.cols - 1));
        int y1 = std::max(0, std::min((int)track.y1, img.rows - 1));
        int x2 = std::max(0, std::min((int)track.x2, img.cols - 1));
        int y2 = std::max(0, std::min((int)track.y2, img.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        const char* label = (track.label >= 0 && track.label < (int)coco_labels.size())
                            ? coco_labels[track.label].c_str() : "unknown";
        if (print_debug) {
            printf("[DEBUG draw]   绘制 ID=%d %s 框: (%d,%d)-(%d,%d)\n",
                   track.track_id, label, x1, y1, x2, y2);
        }

        cv::Scalar color = tracker_color_from_id(track.track_id);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        snprintf(text, sizeof(text), "ID:%d %s %.1f%%", track.track_id, label, track.score * 100.0f);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int tx = x1;
        int ty = y1 - label_size.height - baseLine;
        if (ty < 0) ty = 0;
        if (tx + label_size.width > img.cols) tx = img.cols - label_size.width;

        cv::rectangle(img, cv::Rect(cv::Point(tx, ty),
                      cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);
        cv::putText(img, text, cv::Point(tx, ty + label_size.height),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        const auto& traj = track.trajectory;
        int start_idx = (traj.size() > 20) ? ((int)traj.size() - 20) : 0;
        for (size_t j = (size_t)start_idx + 1; j < traj.size(); j++) {
            int px1 = std::max(0, std::min((int)traj[j - 1].first, img.cols - 1));
            int py1 = std::max(0, std::min((int)traj[j - 1].second, img.rows - 1));
            int px2 = std::max(0, std::min((int)traj[j].first, img.cols - 1));
            int py2 = std::max(0, std::min((int)traj[j].second, img.rows - 1));
            cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2), color, 1);
        }
    }
}

std::string build_html_page() {
    return R"(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 RTSP 控制面板</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; color: #fff; }
        .container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
        h1 { text-align: center; margin-bottom: 30px; color: #00d9ff; font-size: 2em; }
        .card { background: rgba(255,255,255,0.1); border-radius: 16px; padding: 24px; margin-bottom: 20px; backdrop-filter: blur(10px); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .status-item { background: rgba(0,0,0,0.3); border-radius: 12px; padding: 20px; text-align: center; }
        .status-item .label { color: #888; font-size: 0.9em; margin-bottom: 8px; }
        .status-item .value { font-size: 2em; font-weight: bold; color: #00d9ff; }
        .status-item .value.active { color: #00ff88; }
        .status-item .value.inactive { color: #ff4757; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
        .btn { padding: 12px 24px; border: none; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-start { background: linear-gradient(135deg, #00ff88, #00b894); color: #000; }
        .btn-stop { background: linear-gradient(135deg, #ff4757, #c0392b); color: #fff; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .log-container { background: #0a0a0a; border-radius: 8px; padding: 15px; max-height: 200px; overflow-y: auto; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; }
        .log-line { color: #aaa; margin-bottom: 4px; }
        .log-line .time { color: #666; }
        .log-line .info { color: #00d9ff; }
        .log-line .success { color: #00ff88; }
        .footer { text-align: center; color: #666; margin-top: 30px; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv8 RTSP 控制面板</h1>
        
        <div class="card">
            <div class="status-grid">
                <div class="status-item">
                    <div class="label">运行状态</div>
                    <div class="value" id="status-running">--</div>
                </div>
                <div class="status-item">
                    <div class="label">推理状态</div>
                    <div class="value" id="status-inference">--</div>
                </div>
                <div class="status-item">
                    <div class="label">Cam0 FPS</div>
                    <div class="value" id="status-fps0">0</div>
                </div>
                <div class="status-item">
                    <div class="label">Cam1 FPS</div>
                    <div class="value" id="status-fps1">0</div>
                </div>
            </div>
            
#ifdef USE_RTSP_MPP
            <div class="btn-group">
                <button class="btn btn-start" id="btn-rtsp-start">摄像头 RTSP</button>
                <button class="btn btn-start" id="btn-rtsp-video-start">视频 RTSP</button>
                <button class="btn btn-stop" id="btn-rtsp-stop">停止推流</button>
            </div>
#endif

            <div class="card" style="background: rgba(0,0,0,0.2); margin-top: 15px;">
                <h4 style="margin-bottom: 10px;">本地视频推理</h4>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <input type="text" id="video-path" placeholder="视频路径，如 /home/pi/test.mp4"
                           style="flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #444; background: #1a1a2e; color: #fff;">
                    <label style="display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="video-loop"> 循环
                    </label>
                    <button class="btn btn-start" id="btn-video-start">播放视频</button>
                    <button class="btn btn-stop" id="btn-video-stop">停止</button>
                </div>
                <div id="video-status" style="margin-top: 10px; color: #888; font-size: 0.9em;"></div>
            </div>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">日志</h3>
            <div class="log-container" id="log-container"></div>
        </div>
        
        <div class="footer">
            <p>YOLOv8 RKNN 多线程推理服务</p>
        </div>
    </div>
    
    <script>
        const logContainer = document.getElementById('log-container');
        function addLog(msg, type) {
            const line = document.createElement('div');
            line.className = 'log-line';
            line.innerHTML = '<span class="time">[' + new Date().toLocaleTimeString() + ']</span> <span class="' + type + '">' + msg + '</span>';
            logContainer.appendChild(line);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status-running').textContent = data.running ? '运行中' : '已停止';
                    document.getElementById('status-running').className = 'value ' + (data.running ? 'active' : 'inactive');
                    document.getElementById('status-inference').textContent = data.inference_enabled ? '开启' : '关闭';
                    document.getElementById('status-inference').className = 'value ' + (data.inference_enabled ? 'active' : 'inactive');
                    document.getElementById('status-fps0').textContent = data.fps_cam0 || 0;
                    document.getElementById('status-fps1').textContent = data.fps_cam1 || 0;
                })
                .catch(e => console.error(e));
        }
        
#ifdef USE_RTSP_MPP
        document.getElementById('btn-rtsp-start').onclick = () => fetch('/api/rtsp/start', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'success'));
        document.getElementById('btn-rtsp-video-start').onclick = () => fetch('/api/rtsp/video/start', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'success'));
        document.getElementById('btn-rtsp-stop').onclick = () => fetch('/api/rtsp/stop', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'info'));
#endif

        // 视频控制
        document.getElementById('btn-video-start').onclick = () => {
            const path = document.getElementById('video-path').value;
            const loop = document.getElementById('video-loop').checked;
            if (!path) { alert('请输入视频路径'); return; }
            fetch('/api/video/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path, loop: loop})
            }).then(r => r.json()).then(d => {
                addLog(d.message, 'success');
                document.getElementById('video-status').textContent = '正在播放: ' + path + (loop ? ' (循环)' : '');
            });
        };
        document.getElementById('btn-video-stop').onclick = () => {
            fetch('/api/video/stop', {method: 'POST'}).then(r => r.json()).then(d => {
                addLog(d.message, 'info');
                document.getElementById('video-status').textContent = '已停止视频';
            });
        };
        
        setInterval(updateStatus, 1000);
        updateStatus();
        addLog('控制面板已连接', 'info');
    </script>
</body>
</html>
)";
}

void send_response(int client_fd, const std::string& content, const std::string& content_type, int status_code = 200) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " " << (status_code == 200 ? "OK" : "Bad Request") << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << content.size() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    
    send(client_fd, response.str().c_str(), response.str().size(), 0);
    if (!content.empty()) {
        send(client_fd, content.c_str(), content.size(), 0);
    }
}

void handle_client(int client_fd) {
    char buffer[16384] = {0};
    int n = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (n <= 0) {
        close(client_fd);
        return;
    }

    std::string request(buffer);
    std::istringstream iss(request);
    std::string method, path, version;
    iss >> method >> path >> version;

    std::string route = path;
    size_t query_start = path.find('?');
    if (query_start != std::string::npos) {
        route = path.substr(0, query_start);
    }

    printf("[HTTP] %s %s\n", method.c_str(), route.c_str());

    // 解析 POST body (Content-Length 方式)
    std::string post_body;
    if (method == "POST") {
        size_t header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            size_t content_len_start = request.find("Content-Length:");
            if (content_len_start != std::string::npos && content_len_start < header_end) {
                size_t colon = request.find(":", content_len_start);
                if (colon != std::string::npos) {
                    std::string len_str = request.substr(colon + 1, header_end - colon - 1);
                    int content_len = atoi(len_str.c_str());
                    post_body = request.substr(header_end + 4);
                    // 如果 body 不完整，尝试继续读取
                    while ((int)post_body.size() < content_len) {
                        char extra[4096] = {0};
                        int extra_n = recv(client_fd, extra, sizeof(extra) - 1, 0);
                        if (extra_n <= 0) break;
                        post_body.append(extra, extra_n);
                    }
                }
            }
        }
    }
    
    if (route == "/api/status" && method == "GET") {
        // 添加跟踪状态到响应
        std::string model_path;
        std::string label_path;
        std::string model_error;
        std::string tracker_backend = rknn_lite::get_tracker_backend_name();
        std::string tracker_reid_model = rknn_lite::resolve_tracker_reid_model();
        int tracker_skip_frames = rknn_lite::get_deepsort_skip_frames();
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            model_path = g_model_path;
            label_path = g_label_path;
            model_error = g_model_error;
        }
        std::string active_label_path = label_path.empty() ? DEFAULT_LABEL_PATH : label_path;
        std::ostringstream oss;
        oss << "{";
        oss << "\"running\":" << (g_running ? "true" : "false") << ",";
        oss << "\"inference_enabled\":" << (g_inference_enabled ? "true" : "false") << ",";
        oss << "\"model_loaded\":" << (g_model_loaded.load() ? "true" : "false") << ",";
        oss << "\"model_loading\":" << (g_model_loading.load() ? "true" : "false") << ",";
        oss << "\"model_switching\":" << (g_model_switching.load() ? "true" : "false") << ",";
        oss << "\"active_jobs\":" << g_active_jobs.load() << ",";
        oss << "\"model_path\":\"" << json_escape(model_path) << "\",";
        oss << "\"label_path\":\"" << json_escape(active_label_path) << "\",";
        oss << "\"rtsp_url_cam0\":\"" << json_escape(g_rtsp_url_0) << "\",";
        oss << "\"rtsp_url_cam1\":\"" << json_escape(g_rtsp_url_1) << "\",";
        oss << "\"rtsp_url_video\":\"" << json_escape(g_rtsp_url_video) << "\",";
        oss << "\"model_error\":\"" << json_escape(model_error) << "\",";
        oss << "\"current_camera\":" << g_current_cam.load() << ",";
        oss << "\"fps\":" << (g_cam0_fps.load() + g_cam1_fps.load()) << ",";
        oss << "\"fps_cam0\":" << g_cam0_fps.load() << ",";
        oss << "\"fps_cam1\":" << g_cam1_fps.load() << ",";
        oss << "\"tracker_enabled\":" << (g_tracker_enabled ? "true" : "false") << ",";
        oss << "\"tracker_backend\":\"" << json_escape(tracker_backend) << "\",";
        oss << "\"tracker_reid_model\":\"" << json_escape(tracker_reid_model) << "\",";
        oss << "\"tracker_skip_frames\":" << tracker_skip_frames << ",";
        oss << "\"detection_count_cam0\":" << g_cam0_detection_count.load() << ",";
        oss << "\"detection_count_cam1\":" << g_cam1_detection_count.load() << ",";
        oss << "\"threshold\":" << rknn_lite::get_detection_threshold() << ",";
        oss << "\"video_mode\":" << (g_video_mode ? "true" : "false");
#ifdef USE_RTSP_MPP
        oss << ",\"rtsp_streaming\":" << (g_rtsp_streaming ? "true" : "false");
#else
        oss << ",\"rtsp_streaming\":false";
#endif
        oss << "}";
        send_response(client_fd, oss.str(), "application/json");
    }
    else if (route == "/api/threshold/set" && method == "POST") {
        // 设置检测阈值: /api/threshold/set?value=0.6
        float value = 0.5f;
        size_t pos = path.find("value=");
        if (pos != std::string::npos) {
            value = std::stof(path.substr(pos + 6));
        }
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        g_confidence_threshold.store(value);
        rknn_lite::set_detection_threshold(value);
        send_response(client_fd, build_json_response("success",
            "阈值已设置为: " + std::to_string(value)), "application/json");
    }
    else if (route == "/api/threshold/get" && method == "GET") {
        // 获取当前阈值
        std::ostringstream oss;
        oss << "{\"threshold\":" << rknn_lite::get_detection_threshold() << "}";
        send_response(client_fd, oss.str(), "application/json");
    }
    else if (route == "/api/detection/count" && method == "GET") {
        // 获取指定摄像头的检测数量: /api/detection/count?cam=0
        int cam = 0;
        size_t pos = path.find("cam=");
        if (pos != std::string::npos) {
            cam = std::stoi(path.substr(pos + 4));
        }
        if (cam < 0) cam = 0;
        if (cam >= 2) cam = 0;  // 只支持 0 和 1
        int count = (cam == 0) ? g_cam0_detection_count.load() : g_cam1_detection_count.load();
        std::ostringstream oss;
        oss << "{\"cam\":" << cam << ",\"count\":" << count << "}";
        send_response(client_fd, oss.str(), "application/json");
    }
    else if (route == "/api/frame" && method == "GET") {
        // 解析参数: /api/frame?track=1&redraw=1
        bool want_tracker = g_tracker_enabled.load();
        if (path.find("track=1") != std::string::npos || path.find("track=ON") != std::string::npos) {
            want_tracker = true;
        } else if (path.find("track=0") != std::string::npos || path.find("track=OFF") != std::string::npos) {
            want_tracker = false;
        }
        // 默认不二次绘制：推理线程已经在 ori_img 上画过框，二次叠加会出现“一人双框”
        bool force_redraw = (path.find("redraw=1") != std::string::npos || path.find("overlay=1") != std::string::npos);
        
        cv::Mat frame_copy;
        bool available = false;
        int frame_slot = -1;
        {
            std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
            if (!g_latest_frame.empty()) {
                frame_copy = g_latest_frame.clone();
                available = true;
                frame_slot = g_latest_frame_slot.load();
            }
        }
        
        if (available && !frame_copy.empty()) {
            // 如果启用了跟踪，可选二次绘制（仅用于调试）。默认关闭，避免叠加双框。
            if (want_tracker && force_redraw) {
                std::vector<TrackerResultItem> tracks_for_frame;
                {
                    std::lock_guard<std::mutex> lock(g_tracker_mutex);
                    if (frame_slot >= 0 && frame_slot < (int)g_tracker_results.size()) {
                        tracks_for_frame = g_tracker_results[frame_slot];
                    }
                }
                if (!tracks_for_frame.empty()) {
                    draw_tracker_boxes(frame_copy, tracks_for_frame, frame_slot);
                }

                static int redraw_log_counter = 0;
                if ((redraw_log_counter++ % 120) == 0) {
                    printf("[FrameAPI] redraw=1, slot=%d, tracks=%zu (注意：可能与推理线程已有框叠加)\n",
                           frame_slot, tracks_for_frame.size());
                }
            } else if (want_tracker && !force_redraw) {
                static int skip_redraw_log_counter = 0;
                if ((skip_redraw_log_counter++ % 120) == 0) {
                    printf("[FrameAPI] 跳过二次绘制（避免双框）。如需调试叠加，请使用 /api/frame?track=1&redraw=1\n");
                }
            }
            
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame_copy, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
            std::string content(buf.begin(), buf.end());
            send_response(client_fd, content, "image/jpeg");
        } else {
            send_response(client_fd, build_json_response("error", "暂无帧数据"), "application/json", 503);
        }
    }
    else if (route == "/api/tracker" && method == "GET") {
        // 获取/设置跟踪状态
        send_response(client_fd, build_json_response("success", 
            std::string("跟踪状态: ") + (g_tracker_enabled ? "开启" : "关闭") +
            ", 算法: " + rknn_lite::get_tracker_backend_name()), "application/json");
    }
    else if (route.find("/api/tracker/") == 0 && method == "POST") {
        // /api/tracker/0 关闭跟踪 /api/tracker/1 开启跟踪
        std::string val = route.substr(13);
        bool enable = (val == "1" || val == "on" || val == "ON");
        g_tracker_enabled = enable;
        printf("[Tracker] 跟踪已%s\n", enable ? "开启" : "关闭");
        send_response(client_fd, build_json_response("success", 
            std::string("跟踪已") + (enable ? "开启" : "关闭")), "application/json");
    }
    else if (route == "/api/inference/on" && method == "POST") {
        // /api/inference/on?track=0 或 /api/inference/on?track=1
        // 可选:
        //   /api/inference/on?model=../model/RK3588/yolov8s.rknn
        //   /api/inference/on?labels=/path/to/labels.txt
        if (g_model_switching.load()) {
            send_response(client_fd, build_json_response("error", "模型切换中，请稍后重试"), "application/json", 409);
            return;
        }

        // 默认开启跟踪，除非明确指定 track=0
        bool enable_tracker = true;
        if (path.find("track=0") != std::string::npos || path.find("track=OFF") != std::string::npos) {
            enable_tracker = false;
        }

        std::string requested_model = parse_query_param(path, "model");
        bool has_labels_param = path.find("labels=") != std::string::npos;
        std::string requested_labels = parse_query_param(path, "labels");
        bool has_tracker_param = path.find("tracker=") != std::string::npos;
        std::string requested_tracker = has_tracker_param
            ? parse_query_param(path, "tracker")
            : rknn_lite::get_tracker_backend_name();
        TrackerBackend requested_backend;
        if (!parse_tracker_backend_name(requested_tracker, &requested_backend)) {
            send_response(client_fd,
                build_json_response("error", "无效的 tracker 参数，支持 bytetrack 或 deepsort"),
                "application/json", 400);
            return;
        }
        bool has_reid_param = (path.find("reid_model=") != std::string::npos) ||
                              (path.find("reid=") != std::string::npos);
        std::string requested_reid = parse_query_param(path, "reid_model");
        if (requested_reid.empty() && path.find("reid=") != std::string::npos) {
            requested_reid = parse_query_param(path, "reid");
        }
        bool has_tracker_skip_param = (path.find("tracker_skip=") != std::string::npos) ||
                                      (path.find("deepsort_skip=") != std::string::npos);
        std::string requested_skip_text = parse_query_param(path, "tracker_skip");
        if (requested_skip_text.empty() && path.find("deepsort_skip=") != std::string::npos) {
            requested_skip_text = parse_query_param(path, "deepsort_skip");
        }
        int requested_skip_frames = rknn_lite::get_deepsort_skip_frames();
        if (has_tracker_skip_param) {
            char* endptr = nullptr;
            long parsed = strtol(requested_skip_text.c_str(), &endptr, 10);
            if (requested_skip_text.empty() || endptr == requested_skip_text.c_str() || *endptr != '\0' || parsed < 0) {
                send_response(client_fd,
                    build_json_response("error", "无效的 tracker_skip 参数，必须是大于等于 0 的整数"),
                    "application/json", 400);
                return;
            }
            requested_skip_frames = (int)parsed;
        }
        if (requested_labels == "default" || requested_labels == "DEFAULT") {
            requested_labels.clear();
        }
        if (requested_reid == "default" || requested_reid == "DEFAULT") {
            requested_reid.clear();
        }
        if (!requested_model.empty() && !file_exists(requested_model)) {
            send_response(client_fd,
                build_json_response("error", "模型文件不存在或不可读: " + requested_model),
                "application/json", 400);
            return;
        }
        if (has_labels_param && !requested_labels.empty() && !file_exists(requested_labels)) {
            send_response(client_fd,
                build_json_response("error", "标签文件不存在或不可读: " + requested_labels),
                "application/json", 400);
            return;
        }
        if (has_reid_param && !requested_reid.empty() && !file_exists(requested_reid)) {
            send_response(client_fd,
                build_json_response("error", "DeepSORT ReID 模型不存在或不可读: " + requested_reid),
                "application/json", 400);
            return;
        }
        if (requested_backend == TrackerBackend::DeepSort) {
            std::string resolved_reid = has_reid_param ? requested_reid : rknn_lite::resolve_tracker_reid_model();
            if (resolved_reid.empty()) {
                send_response(client_fd,
                    build_json_response("error", "未找到可用的 DeepSORT ReID 模型，请传入 reid_model 参数"),
                    "application/json", 400);
                return;
            }
            if (!file_exists(resolved_reid)) {
                send_response(client_fd,
                    build_json_response("error", "DeepSORT ReID 模型不存在或不可读: " + resolved_reid),
                    "application/json", 400);
                return;
            }
        }

        if (!requested_model.empty() || has_labels_param || has_tracker_param || has_reid_param || has_tracker_skip_param) {
            std::string current_model;
            std::string current_labels;
            std::string current_tracker;
            std::string current_reid;
            bool loaded = false;
            {
                std::lock_guard<std::mutex> lock(g_model_mutex);
                loaded = g_model_loaded.load();
                current_model = g_model_path;
                current_labels = g_label_path;
            }
            current_tracker = rknn_lite::get_tracker_backend_name();
            current_reid = rknn_lite::get_tracker_reid_model_override();
            int current_skip = rknn_lite::get_deepsort_skip_frames();

            bool model_changed = (!requested_model.empty() && requested_model != current_model);
            bool labels_changed = (has_labels_param && requested_labels != current_labels);
            bool tracker_changed = (has_tracker_param && requested_tracker != current_tracker);
            bool reid_changed = (has_reid_param && requested_reid != current_reid);
            bool skip_changed = (has_tracker_skip_param && requested_skip_frames != current_skip);

            if (loaded && (model_changed || labels_changed || tracker_changed || reid_changed)) {
                if (g_inference_enabled.load()) {
                    send_response(client_fd,
                        build_json_response("error", "请先调用 /api/inference/off 关闭推理后再切换模型/标签/跟踪算法"),
                        "application/json", 409);
                    return;
                }
                std::string unload_msg;
                if (!unload_model_runtime(&unload_msg)) {
                    send_response(client_fd, build_json_response("error", unload_msg), "application/json", 500);
                    return;
                }
            }

            {
                std::lock_guard<std::mutex> lock(g_model_mutex);
                if (!requested_model.empty()) {
                    g_model_path = requested_model;
                }
                if (has_labels_param) {
                    g_label_path = requested_labels;
                }
                g_model_error.clear();
            }
            if (tracker_changed) {
                rknn_lite::set_tracker_backend(requested_tracker);
            }
            if (reid_changed) {
                rknn_lite::set_tracker_reid_model_override(requested_reid);
            }
            if (skip_changed) {
                rknn_lite::set_deepsort_skip_frames(requested_skip_frames);
            }
        }

        std::string label_override;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            label_override = g_label_path;
        }
        rknn_lite::set_label_file_override(label_override);

        std::string load_msg;
        if (!g_model_loaded.load()) {
#ifdef USE_RTSP_MPP
            stop_video_rtsp_raw_thread();
#endif
            if (!ensure_model_loaded(&load_msg)) {
                g_inference_enabled = false;
                g_tracker_enabled = false;
                send_response(client_fd, build_json_response("error", load_msg), "application/json", 500);
                return;
            }
        }

        std::string active_model;
        std::string active_labels;
        std::string active_tracker = rknn_lite::get_tracker_backend_name();
        std::string active_reid = rknn_lite::resolve_tracker_reid_model();
        int active_skip_frames = rknn_lite::get_deepsort_skip_frames();
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            active_model = g_model_path;
            active_labels = g_label_path.empty() ? DEFAULT_LABEL_PATH : g_label_path;
        }

        g_inference_enabled = true;
        g_tracker_enabled = enable_tracker;
        printf("[Inference] 推理已开启, 跟踪: %s, 算法: %s, skip=%d, 模型: %s, 标签: %s, ReID: %s\n",
               enable_tracker ? "开启" : "关闭",
               active_tracker.c_str(),
               active_skip_frames,
               active_model.c_str(),
               active_labels.c_str(),
               active_reid.empty() ? "(none)" : active_reid.c_str());
        send_response(client_fd, build_json_response("success", 
            std::string("推理已开启, 跟踪: ") + (enable_tracker ? "开启" : "关闭") +
            ", 算法: " + active_tracker +
            ", skip: " + std::to_string(active_skip_frames) +
            ", 模型: " + active_model +
            ", 标签: " + active_labels +
            (active_tracker == "deepsort" ? ", ReID: " + active_reid : "")), "application/json");
    }
    else if (route == "/api/inference/off" && method == "POST") {
        bool unload_requested =
            (path.find("unload=1") != std::string::npos) ||
            (path.find("unload=true") != std::string::npos) ||
            (path.find("unload=ON") != std::string::npos);
        std::string requested_model = parse_query_param(path, "model");
        bool has_labels_param = path.find("labels=") != std::string::npos;
        std::string requested_labels = parse_query_param(path, "labels");
        if (requested_labels == "default" || requested_labels == "DEFAULT") {
            requested_labels.clear();
        }

        if (!requested_model.empty() && !file_exists(requested_model)) {
            send_response(client_fd,
                build_json_response("error", "模型文件不存在或不可读: " + requested_model),
                "application/json", 400);
            return;
        }
        if (has_labels_param && !requested_labels.empty() && !file_exists(requested_labels)) {
            send_response(client_fd,
                build_json_response("error", "标签文件不存在或不可读: " + requested_labels),
                "application/json", 400);
            return;
        }

        g_inference_enabled = false;
        g_tracker_enabled = false;
        // 清空跟踪结果，停止显示检测框
        {
            std::lock_guard<std::mutex> lock(g_tracker_mutex);
            for (auto& tracks : g_tracker_results) {
                tracks.clear();
            }
        }

        std::string message = "推理已关闭";
        if (unload_requested || !requested_model.empty() || has_labels_param) {
            std::string unload_msg;
            if (!unload_model_runtime(&unload_msg)) {
                send_response(client_fd, build_json_response("error", unload_msg), "application/json", 500);
                return;
            }
            message = "推理已关闭，模型已卸载";
        }

        if (!requested_model.empty()) {
            {
                std::lock_guard<std::mutex> lock(g_model_mutex);
                g_model_path = requested_model;
                g_model_error.clear();
            }
            message += "，下次将加载: " + requested_model;
        }
        if (has_labels_param) {
            std::string active_label;
            {
                std::lock_guard<std::mutex> lock(g_model_mutex);
                g_label_path = requested_labels;
                g_model_error.clear();
                active_label = g_label_path.empty() ? DEFAULT_LABEL_PATH : g_label_path;
            }
            rknn_lite::set_label_file_override(requested_labels);
            message += "，标签: " + active_label;
        }

        printf("[Inference] 推理已关闭\n");
        send_response(client_fd, build_json_response("success", message), "application/json");
    }
    else if (route.find("/api/camera/") == 0 && method == "POST") {
        int cam = std::stoi(route.substr(12));
        if (cam >= 0 && cam <= 2) {
            g_current_cam = cam;
            send_response(client_fd, build_json_response("success", "已切换到摄像头 " + std::to_string(cam)), "application/json");
        } else {
            send_response(client_fd, build_json_response("error", "无效的摄像头ID"), "application/json", 400);
        }
    }
#ifdef USE_RTSP_MPP
    else if (route == "/api/rtsp/start" && method == "POST") {
        std::lock_guard<std::mutex> lock(g_rtsp_mutex);
        printf("[RTSP] /api/rtsp/start 请求: streaming=%d video_mode=%d sender0=%p sender1=%p senderV=%p rss=%ldKB\n",
               g_rtsp_streaming.load() ? 1 : 0,
               g_video_mode.load() ? 1 : 0,
               (void*)g_rtsp_sender0, (void*)g_rtsp_sender1, (void*)g_rtsp_sender_video,
               get_process_rss_kb());
        if (g_video_mode.load() && g_rtsp_streaming.load() &&
            g_rtsp_sender_video && g_rtsp_sender_video->inited()) {
            send_response(client_fd, build_json_response("success", "视频 RTSP 已在运行"), "application/json");
            return;
        }
        if (!g_video_mode.load() && g_rtsp_streaming.load() &&
            g_rtsp_sender0 && g_rtsp_sender0->inited() &&
            g_rtsp_sender1 && g_rtsp_sender1->inited()) {
            send_response(client_fd, build_json_response("success", "摄像头 RTSP 已在运行"), "application/json");
            return;
        }

        // 每次从停止态重新开启时，先重建推流器，避免复用已断开的 socket
        if (!g_rtsp_streaming.load()) {
            printf("[RTSP] 重新启动前清理旧 sender\n");
            stop_rtsp_senders_locked();
        }
        
        if (g_video_mode.load()) {
            // 视频模式：创建视频推流器
            int vw = g_video_width.load();
            int vh = g_video_height.load();
            int vf = g_video_fps.load();
            {
                std::lock_guard<std::mutex> video_lock(g_video_mutex);
                if (g_video_running.load() && g_video_mode.load()) {
                    (void)ensure_video_source_open_locked();
                }
#ifdef USE_RTSP_MPP
                if (g_video_hw_enabled.load() && g_video_hw_reader && g_video_hw_reader->IsOpen()) {
                    if (g_video_hw_reader->Width() > 0) vw = g_video_hw_reader->Width();
                    if (g_video_hw_reader->Height() > 0) vh = g_video_hw_reader->Height();
                    if (g_video_hw_reader->Fps() > 0) vf = (int)(g_video_hw_reader->Fps() + 0.5);
                } else
#endif
                if (g_video_cap.isOpened()) {
                    int cap_w = (int)g_video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
                    int cap_h = (int)g_video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                    int cap_fps = (int)(g_video_cap.get(cv::CAP_PROP_FPS) + 0.5);
                    if (cap_w > 0) vw = cap_w;
                    if (cap_h > 0) vh = cap_h;
                    if (cap_fps > 0) vf = cap_fps;
                }
            }
            if (!g_rtsp_sender_video || !g_rtsp_sender_video->inited()) {
                if (g_rtsp_sender_video) {
                    delete g_rtsp_sender_video;
                    g_rtsp_sender_video = nullptr;
                }
                g_rtsp_sender_video = new RtspMppSender();
                if (!g_rtsp_sender_video->init(g_rtsp_url_video.c_str(), vw, vh, vf > 0 ? vf : 25)) {
                    delete g_rtsp_sender_video;
                    g_rtsp_sender_video = nullptr;
                    printf("[RTSP] 视频推流启动失败\n");
                    send_response(client_fd, build_json_response("error", "视频 RTSP 启动失败"), "application/json", 500);
                    return;
                }
                printf("[RTSP] 视频推流已启动 (%dx%d @ %dfps) -> %s\n", vw, vh, vf > 0 ? vf : 25, g_rtsp_url_video.c_str());
            }
            g_rtsp_streaming = true;
            send_response(client_fd, build_json_response("success", "视频 RTSP 推流已启动"), "application/json");
            return;
        } else {
            // 摄像头模式：创建摄像头推流器
            // 摄像头采集端设置为 30fps，这里保持一致，避免推送时基与实际送帧速率不一致导致播放器卡顿感。
            int cam0_w = 640, cam0_h = 480, cam0_fps = 30;
            int cam1_w = 640, cam1_h = 480, cam1_fps = 30;
            bool cam0_ok = (g_rtsp_sender0 != nullptr && g_rtsp_sender0->inited());
            bool cam1_ok = (g_rtsp_sender1 != nullptr && g_rtsp_sender1->inited());
            if (!cam0_ok) {
                if (g_rtsp_sender0) {
                    delete g_rtsp_sender0;
                    g_rtsp_sender0 = nullptr;
                }
                g_rtsp_sender0 = new RtspMppSender();
                if (!g_rtsp_sender0->init(g_rtsp_url_0.c_str(), cam0_w, cam0_h, cam0_fps)) {
                    delete g_rtsp_sender0;
                    g_rtsp_sender0 = nullptr;
                    printf("[RTSP] Cam0 启动失败: %s\n", g_rtsp_url_0.c_str());
                } else {
                    printf("[RTSP] Cam0 已启动 (%dx%d @ %dfps)\n", cam0_w, cam0_h, cam0_fps);
                    cam0_ok = true;
                }
            }
            if (!cam1_ok) {
                if (g_rtsp_sender1) {
                    delete g_rtsp_sender1;
                    g_rtsp_sender1 = nullptr;
                }
                g_rtsp_sender1 = new RtspMppSender();
                if (!g_rtsp_sender1->init(g_rtsp_url_1.c_str(), cam1_w, cam1_h, cam1_fps)) {
                    delete g_rtsp_sender1;
                    g_rtsp_sender1 = nullptr;
                    printf("[RTSP] Cam1 启动失败: %s\n", g_rtsp_url_1.c_str());
                } else {
                    printf("[RTSP] Cam1 已启动 (%dx%d @ %dfps)\n", cam1_w, cam1_h, cam1_fps);
                    cam1_ok = true;
                }
            }
            if (!cam0_ok && !cam1_ok) {
                g_rtsp_streaming = false;
                send_response(client_fd,
                    build_json_response("error", "RTSP 推流启动失败（cam0/cam1 均未连接）"),
                    "application/json", 500);
                return;
            }
            g_rtsp_streaming = true;
            if (g_model_loaded.load()) {
                printf("[RTSP] /api/rtsp/start 成功: cam0=%d cam1=%d rss=%ldKB\n",
                       cam0_ok ? 1 : 0, cam1_ok ? 1 : 0, get_process_rss_kb());
                send_response(client_fd,
                    build_json_response("success", "RTSP 推流已启动"),
                    "application/json");
            } else {
                printf("[RTSP] /api/rtsp/start 成功(裸流): cam0=%d cam1=%d rss=%ldKB\n",
                       cam0_ok ? 1 : 0, cam1_ok ? 1 : 0, get_process_rss_kb());
                send_response(client_fd,
                    build_json_response("success", "RTSP 裸流已启动（未开启推理）"),
                    "application/json");
            }
            return;
        }
    }
    else if (route == "/api/rtsp/video/start" && method == "POST") {
        // 视频模式专用推流，使用原始视频分辨率
        std::lock_guard<std::mutex> lock(g_rtsp_mutex);
        if (g_rtsp_streaming.load() && g_rtsp_sender_video && g_rtsp_sender_video->inited()) {
            send_response(client_fd, build_json_response("success", "视频 RTSP 已在运行"), "application/json");
            return;
        }
        if (!g_rtsp_streaming.load()) {
            stop_rtsp_senders_locked();
        }
        int vw = g_video_width.load();
        int vh = g_video_height.load();
        int vf = g_video_fps.load();
        {
            std::lock_guard<std::mutex> video_lock(g_video_mutex);
            if (g_video_running.load() && g_video_mode.load()) {
                (void)ensure_video_source_open_locked();
            }
#ifdef USE_RTSP_MPP
            if (g_video_hw_enabled.load() && g_video_hw_reader && g_video_hw_reader->IsOpen()) {
                if (g_video_hw_reader->Width() > 0) vw = g_video_hw_reader->Width();
                if (g_video_hw_reader->Height() > 0) vh = g_video_hw_reader->Height();
                if (g_video_hw_reader->Fps() > 0) vf = (int)(g_video_hw_reader->Fps() + 0.5);
            } else
#endif
            if (g_video_cap.isOpened()) {
                int cap_w = (int)g_video_cap.get(cv::CAP_PROP_FRAME_WIDTH);
                int cap_h = (int)g_video_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                int cap_fps = (int)(g_video_cap.get(cv::CAP_PROP_FPS) + 0.5);
                if (cap_w > 0) vw = cap_w;
                if (cap_h > 0) vh = cap_h;
                if (cap_fps > 0) vf = cap_fps;
            }
        }
        if (!g_rtsp_sender_video || !g_rtsp_sender_video->inited()) {
            if (g_rtsp_sender_video) {
                delete g_rtsp_sender_video;
                g_rtsp_sender_video = nullptr;
            }
            g_rtsp_sender_video = new RtspMppSender();
            if (!g_rtsp_sender_video->init(g_rtsp_url_video.c_str(), vw, vh, vf > 0 ? vf : 25)) {
                delete g_rtsp_sender_video;
                g_rtsp_sender_video = nullptr;
                printf("[RTSP] 视频推流启动失败\n");
                send_response(client_fd, build_json_response("error", "视频 RTSP 启动失败"), "application/json", 500);
                return;
            }
            printf("[RTSP] 视频推流已启动 (%dx%d @ %dfps)\n", vw, vh, vf > 0 ? vf : 25);
        }
        g_rtsp_streaming = true;
        if (!g_model_loaded.load()) {
            start_video_rtsp_raw_thread_if_needed();
        }
        send_response(client_fd, build_json_response("success",
            "视频推流已启动 (" + std::to_string(vw) + "x" + std::to_string(vh) + ")"), "application/json");
    }
    else if (route == "/api/rtsp/stop" && method == "POST") {
        g_video_rtsp_raw_stop = true;
        {
            std::lock_guard<std::mutex> lock(g_rtsp_mutex);
            bool was_streaming = g_rtsp_streaming.load();
            g_rtsp_streaming = false;
            stop_rtsp_senders_locked();
            printf("[RTSP] /api/rtsp/stop: was_streaming=%d, sender 已销毁, rss=%ldKB\n",
                   was_streaming ? 1 : 0, get_process_rss_kb());
        }
        stop_video_rtsp_raw_thread();
        send_response(client_fd, build_json_response("success", "RTSP 推流已停止"), "application/json");
    }
#endif
    else if (route == "/api/video/status" && method == "GET") {
        std::ostringstream oss;
        oss << "{";
        oss << "\"video_mode\":" << (g_video_mode ? "true" : "false") << ",";
        oss << "\"video_path\":\"" << g_video_path << "\",";
        oss << "\"video_loop\":" << (g_video_loop ? "true" : "false");
        oss << "}";
        send_response(client_fd, oss.str(), "application/json");
    }
    else if (route == "/api/video/start" && method == "POST") {
        // 读取 POST body
        std::string path;
        bool loop = false;
        parse_json_body(post_body, path, loop);

        if (path.empty()) {
            send_response(client_fd, build_json_response("error", "缺少 path 参数"), "application/json", 400);
        } else {
            // 防抖：相同视频重复 start 时不重启，避免频繁 reset 导致不稳定
            bool already_running_same_video = false;
            {
                std::lock_guard<std::mutex> lock(g_video_mutex);
                already_running_same_video =
                    g_video_mode.load() &&
                    g_video_running.load() &&
                    (g_video_path == path) &&
                    (g_video_loop.load() == loop);
            }
            if (already_running_same_video) {
                send_response(client_fd, build_json_response("success",
                    "视频已在播放: " + path + (loop ? " (循环)" : "")), "application/json");
                return;
            }

            start_video_reader(path, loop);
            printf("[Video] 开始播放视频: %s (loop=%s)\n", path.c_str(), loop ? "true" : "false");
            send_response(client_fd, build_json_response("success",
                "视频已启动: " + path + (loop ? " (循环)" : "")), "application/json");
        }
    }
    else if (route == "/api/video/stop" && method == "POST") {
        stop_video_reader();
        printf("[Video] 已停止视频，恢复摄像头\n");
        send_response(client_fd, build_json_response("success", "已停止视频，恢复摄像头"), "application/json");
    }
    else if (route == "/" || route == "/index.html") {
        send_response(client_fd, build_html_page(), "text/html");
    }
    else {
        send_response(client_fd, build_json_response("error", "Unknown endpoint"), "application/json", 404);
    }
    
    close(client_fd);
}

void http_server_thread() {
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        printf("[HTTP] socket 失败\n");
        return;
    }
    
    int opt = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(g_http_port);
    
    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("[HTTP] bind 端口 %d 失败\n", g_http_port);
        close(g_server_fd);
        g_server_fd = -1;
        return;
    }
    
    struct timeval tv = {0, 100000};
    setsockopt(g_server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    listen(g_server_fd, 10);
    printf("[HTTP] HTTP 服务器已启动: http://0.0.0.0:%d\n", g_http_port);
    
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd >= 0) {
            std::thread(handle_client, client_fd).detach();
        }
    }
    
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
    printf("[HTTP] HTTP 服务器已停止\n");
}

void signal_handler(int sig) {
    printf("\n[Main] 收到信号 %d, 正在停止...\n", sig);
    g_running = false;
    g_inference_enabled = false;
    g_tracker_enabled = false;
    g_model_switching = true;

    if (g_server_fd >= 0) {
        shutdown(g_server_fd, SHUT_RDWR);
        close(g_server_fd);
        g_server_fd = -1;
    }

#ifdef USE_RTSP_MPP
    g_rtsp_streaming = false;
    g_video_rtsp_raw_stop = true;
#endif

    g_video_mode = false;
    g_video_running = false;
}

void print_banner() {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║        YOLOv8 RKNN HTTP 控制服务 (多线程推理版)         ║\n");
    printf("║        OrangePi 5B / RK3588                               ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  HTTP 端口: %d                                            ║\n", g_http_port);
    printf("║  模型加载: 首次调用 /api/inference/on 时自动加载          ║\n");
    printf("║  每摄像头 slot: %d                                         ║\n", SLOTS_PER_CAM);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 读取 RTSP 环境变量（可选）
    const char* env_rtsp_host = getenv("RTSP_HOST");
    const char* env_rtsp_port = getenv("RTSP_PORT");
    const char* env_tracker_backend = getenv("TRACKER_BACKEND");
    const char* env_tracker_reid = getenv("TRACKER_REID_MODEL");
    if (env_rtsp_host && *env_rtsp_host) {
        g_rtsp_host = env_rtsp_host;
    }
    if (env_rtsp_port && *env_rtsp_port) {
        int p = atoi(env_rtsp_port);
        if (p > 0 && p <= 65535) g_rtsp_port = p;
    }
    if (env_tracker_backend && *env_tracker_backend) {
        if (!rknn_lite::set_tracker_backend(env_tracker_backend)) {
            printf("[Tracker] 警告: 无效的 TRACKER_BACKEND=%s，继续使用默认 bytetrack\n", env_tracker_backend);
        }
    }
    if (env_tracker_reid && *env_tracker_reid) {
        rknn_lite::set_tracker_reid_model_override(env_tracker_reid);
    }

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            g_http_port = std::stoi(argv[++i]);
        } else if (arg == "--rtsp-host" && i + 1 < argc) {
            g_rtsp_host = argv[++i];
        } else if (arg == "--rtsp-port" && i + 1 < argc) {
            int p = std::stoi(argv[++i]);
            if (p > 0 && p <= 65535) {
                g_rtsp_port = p;
            }
        } else if (arg == "--tracker-backend" && i + 1 < argc) {
            if (!rknn_lite::set_tracker_backend(argv[++i])) {
                printf("[Tracker] 警告: 无效的 --tracker-backend，继续使用默认 bytetrack\n");
            }
        } else if (arg == "--reid-model" && i + 1 < argc) {
            rknn_lite::set_tracker_reid_model_override(argv[++i]);
        }
    }

    refresh_rtsp_urls();
    print_banner();
    printf("[RTSP] 推流目标: %s | %s | %s\n",
           g_rtsp_url_0.c_str(), g_rtsp_url_1.c_str(), g_rtsp_url_video.c_str());
    printf("[Tracker] 默认算法: %s | ReID: %s\n",
           rknn_lite::get_tracker_backend_name().c_str(),
           rknn_lite::resolve_tracker_reid_model().empty() ? "(none)" : rknn_lite::resolve_tracker_reid_model().c_str());
    
    // 初始化 slot 容器，模型在 /api/inference/on 时懒加载
    int total_slots = SLOTS_PER_CAM * 2;
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        g_rkpool.assign(total_slots, nullptr);
    }
    auto& rkpool = g_rkpool;
    
    // 打开摄像头：优先 DMABUF，失败时回退到 OpenCV V4L2
    v4l2_dmabuf::CaptureContext dmabuf_cap0;
    v4l2_dmabuf::CaptureContext dmabuf_cap1;
    cv::VideoCapture cap0;
    cv::VideoCapture cap1;

    v4l2_dmabuf::CaptureConfig cap0_cfg;
    cap0_cfg.device_name = "/dev/video0";
    cap0_cfg.width = 640;
    cap0_cfg.height = 480;
    cap0_cfg.fps = 30;
    cap0_cfg.dma_buffers = 4;

    std::string cap0_err;
    if (v4l2_dmabuf::open_capture(cap0_cfg, &dmabuf_cap0, &cap0_err) == 0) {
        printf("[Capture] 摄像头0 DMABUF 已打开: %s (%dx%d @ %dfps)\n",
               cap0_cfg.device_name.c_str(), dmabuf_cap0.width, dmabuf_cap0.height, cap0_cfg.fps);
    } else {
        cap0.open(0, cv::CAP_V4L2);
        if (cap0.isOpened()) {
            cap0.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap0.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap0.set(cv::CAP_PROP_FPS, 30);
            printf("[Capture] 摄像头0 OpenCV 回退已打开 (DMABUF失败: %s)\n", cap0_err.c_str());
        } else {
            printf("[Capture] 警告: 无法打开摄像头0 (DMABUF失败: %s)\n", cap0_err.c_str());
        }
    }

    v4l2_dmabuf::CaptureConfig cap1_cfg;
    cap1_cfg.device_name = "/dev/video2";
    cap1_cfg.width = 640;
    cap1_cfg.height = 480;
    cap1_cfg.fps = 30;
    cap1_cfg.dma_buffers = 4;

    std::string cap1_err;
    if (v4l2_dmabuf::open_capture(cap1_cfg, &dmabuf_cap1, &cap1_err) == 0) {
        printf("[Capture] 摄像头1 DMABUF 已打开: %s (%dx%d @ %dfps)\n",
               cap1_cfg.device_name.c_str(), dmabuf_cap1.width, dmabuf_cap1.height, cap1_cfg.fps);
    } else {
        cap1.open(2, cv::CAP_V4L2);
        if (cap1.isOpened()) {
            cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap1.set(cv::CAP_PROP_FPS, 30);
            printf("[Capture] 摄像头1 OpenCV 回退已打开 (DMABUF失败: %s)\n", cap1_err.c_str());
        } else {
            printf("[Capture] 警告: 无法打开摄像头1 (DMABUF失败: %s)\n", cap1_err.c_str());
        }
    }
    
    // 初始化 slot 状态
    struct SlotState {
        std::atomic<bool> busy{false};
        std::atomic<bool> ready{false};
        std::thread::id thread_id;
    };
    
    std::vector<SlotState> slot_states(total_slots);
    
    // 初始化跟踪结果向量
    g_tracker_results.resize(total_slots);
    
    // FPS 统计
    int frames0 = 0, frames1 = 0;
    int push0 = 0, push1 = 0;
    bool video_raw_pace_initialized = false;
    auto video_raw_next_deadline = std::chrono::steady_clock::now();
    struct timeval time_start, time_now;
    gettimeofday(&time_start, nullptr);
    long last_fps_time_ms = time_start.tv_sec * 1000 + time_start.tv_usec / 1000;
    // 启动 HTTP 服务器
    std::thread http_thread(http_server_thread);
    
    printf("[Main] 流水线已启动\n");
    
    // 主循环
    while (g_running) {
        if (g_model_switching.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        if (!g_model_loaded.load()) {
#ifdef USE_RTSP_MPP
            // 未加载模型时，仍允许 RTSP 裸流（不做推理）
            if (g_rtsp_streaming.load()) {
                if (g_video_mode.load()) {
                    if (!g_video_rtsp_raw_running.load()) {
                        start_video_rtsp_raw_thread_if_needed();
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                    continue;
                }

                video_raw_pace_initialized = false;

                bool got_frame = false;

                {
                    cv::Mat raw0;
                    if (read_camera_frame(&dmabuf_cap0, &cap0, &raw0) && !raw0.empty()) {
                        got_frame = true;
                        frames0++;
                        {
                            std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
                            g_latest_frame = raw0.clone();
                            g_frame_available = true;
                            g_latest_frame_slot = 0;
                        }
                        {
                            std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                            if (g_rtsp_sender0 && g_rtsp_sender0->inited()) {
                                if (g_rtsp_sender0->push(raw0)) {
                                    push0++;
                                } else {
                                    printf("[RTSP] Cam0 裸流写包失败，已自动停止\n");
                                    g_rtsp_streaming = false;
                                }
                            }
                        }
                    }
                }

                {
                    cv::Mat raw1;
                    if (read_camera_frame(&dmabuf_cap1, &cap1, &raw1) && !raw1.empty()) {
                        got_frame = true;
                        frames1++;
                        {
                            std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
                            g_latest_frame = raw1.clone();
                            g_frame_available = true;
                            g_latest_frame_slot = SLOTS_PER_CAM;
                        }
                        {
                            std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                            if (g_rtsp_sender1 && g_rtsp_sender1->inited()) {
                                if (g_rtsp_sender1->push(raw1)) {
                                    push1++;
                                } else {
                                    printf("[RTSP] Cam1 裸流写包失败，已自动停止\n");
                                    g_rtsp_streaming = false;
                                }
                            }
                        }
                    }
                }

                if (!got_frame) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }

                gettimeofday(&time_now, nullptr);
                long now_ms = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
                if (now_ms - last_fps_time_ms >= 1000) {
                    float elapsed = (now_ms - last_fps_time_ms) / 1000.0f;
                    g_cam0_fps.store((int)(frames0 / elapsed));
                    g_cam1_fps.store((int)(frames1 / elapsed));
                    if (push0 > 0 || push1 > 0) {
                        printf("[FPS-RAW] Cam0: %d FPS | Cam1: %d FPS | RTSP: %d | %d\n",
                               g_cam0_fps.load(), g_cam1_fps.load(), push0, push1);
                    }
                    frames0 = 0;
                    frames1 = 0;
                    push0 = 0;
                    push1 = 0;
                    last_fps_time_ms = now_ms;
                }
                continue;
            }
#endif
            g_cam0_fps.store(0);
            g_cam1_fps.store(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        bool any_done = false;
        
        // 检查并处理完成的 slot
        for (int i = 0; i < total_slots; i++) {
            if (g_model_switching.load()) break;
            int cam = (i < SLOTS_PER_CAM) ? 0 : 1;
            
            if (slot_states[i].ready.load()) {
                any_done = true;
                cv::Mat worker_frame;
                {
                    std::lock_guard<std::mutex> lock(g_model_mutex);
                    if (i < (int)rkpool.size() && rkpool[i]) {
                        worker_frame = rkpool[i]->ori_img.clone();
                    }
                }
                if (worker_frame.empty()) {
                    slot_states[i].ready.store(false);
                    continue;
                }
                
                // 更新最新帧（供 HTTP API 获取）
                // 视频模式只用 slot 0，摄像头模式用各自摄像头
                bool should_update_frame = false;
                if (g_video_mode.load()) {
                    should_update_frame = (i == 0);  // 视频模式只用 slot 0
                } else {
                    should_update_frame = true;  // 摄像头模式更新各自
                }
                
                if (should_update_frame) {
                    std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
                    g_latest_frame = worker_frame.clone();
                    g_frame_available = true;
                    g_latest_frame_slot = i;
                }
                
                // RTSP 推流（绘制跟踪框）
#ifdef USE_RTSP_MPP
                if (g_rtsp_streaming.load()) {
                    cv::Mat frame_for_rtsp = worker_frame.clone();

                    bool use_tracker = g_tracker_enabled.load();
                    if (use_tracker) {
                        // 注意：interf_detect_only() 已在 ori_img 上绘制跟踪框
                        // 这里若再次绘制会导致 RTSP 上出现“一人双框”
                        static int rtsp_skip_redraw_log_counter = 0;
                        if ((rtsp_skip_redraw_log_counter++ % 300) == 0) {
                            printf("[RTSP] 跳过二次绘制（避免双框）\n");
                        }
                    }

                    {
                        std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                        if (g_video_mode.load() && g_rtsp_sender_video) {
                            if (g_rtsp_sender_video->push(frame_for_rtsp)) {
                                push0++;
                            } else {
                                printf("[RTSP] 视频推流写包失败，已自动停止，请重新调用 /api/rtsp/video/start\n");
                                g_rtsp_streaming = false;
                            }
                        } else if (!g_video_mode.load()) {
                            if (cam == 0 && g_rtsp_sender0) {
                                if (g_rtsp_sender0->push(frame_for_rtsp)) {
                                    push0++;
                                } else {
                                    printf("[RTSP] Cam0 推流写包失败，已自动停止，请重新调用 /api/rtsp/start\n");
                                    g_rtsp_streaming = false;
                                }
                            }
                            if (cam == 1 && g_rtsp_sender1) {
                                if (g_rtsp_sender1->push(frame_for_rtsp)) {
                                    push1++;
                                } else {
                                    printf("[RTSP] Cam1 推流写包失败，已自动停止，请重新调用 /api/rtsp/start\n");
                                    g_rtsp_streaming = false;
                                }
                            }
                        }
                    }
                }
#endif
                
                if (cam == 0) frames0++;
                else frames1++;
                
                slot_states[i].ready.store(false);
            }
        }
        
        // 给空闲的 slot 分配新任务
        for (int i = 0; i < total_slots; i++) {
            if (g_model_switching.load()) break;
            if (slot_states[i].busy.load() || slot_states[i].ready.load()) continue;

            int cam = (i < SLOTS_PER_CAM) ? 0 : 1;
            rknn_lite* worker = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_model_mutex);
                if (i < (int)rkpool.size()) {
                    worker = rkpool[i];
                }
            }
            if (!worker) continue;

            cv::Mat frame;
            bool frame_ok = false;
            bool do_inference = g_inference_enabled.load();
            CameraDmabufFrameInfo camera_frame_info;

            // 检查是否是视频文件模式
            bool video_mode = g_video_mode.load();
            if (video_mode) {
                std::lock_guard<std::mutex> lock(g_video_mutex);
                VideoFrameInfo video_frame_info;
                if (!ensure_video_source_open_locked()) {
                        printf("[Video] 无法打开视频: %s\n", g_video_path.c_str());
                        g_video_running = false;
                        g_video_mode = false;
                }
                if (g_video_running.load() && g_video_mode.load()) {
                    frame_ok = read_video_frame_locked(frame, &video_frame_info);
                }
                if (!frame_ok) {
                    close_video_source_locked();
                    g_video_running = false;
                    g_video_mode = false;
                    printf("[Video] 视频播放完毕\n");
                } else if (video_frame_info.valid) {
                    if (i >= 0 && i < (int)rkpool.size() && rkpool[i]) {
                        rkpool[i]->set_video_dmabuf_frame(video_frame_info.fd, video_frame_info.size,
                                                          video_frame_info.width, video_frame_info.height,
                                                          video_frame_info.wstride, video_frame_info.hstride,
                                                          video_frame_info.drm_format);
                        video_frame_info.fd = -1;
                    }
                } else if (video_frame_info.nv12_valid) {
                    if (i >= 0 && i < (int)rkpool.size() && rkpool[i]) {
                        rkpool[i]->set_video_nv12_frame(video_frame_info.nv12_packed,
                                                        video_frame_info.width, video_frame_info.height,
                                                        video_frame_info.wstride, video_frame_info.hstride);
                    }
                }
            } else {
                // 使用摄像头
                v4l2_dmabuf::CaptureContext* active_cap = (cam == 0) ? &dmabuf_cap0 : &dmabuf_cap1;
                cv::VideoCapture* active_cv_cap = (cam == 0) ? &cap0 : &cap1;
                frame_ok = acquire_camera_frame(active_cap, active_cv_cap, &frame, &camera_frame_info);
                if (frame_ok && do_inference && camera_frame_info.valid) {
                    (void)worker->prepare_camera_dmabuf_input(camera_frame_info.fd,
                                                              camera_frame_info.width,
                                                              camera_frame_info.height,
                                                              camera_frame_info.wstride,
                                                              camera_frame_info.hstride,
                                                              camera_frame_info.rga_format);
                }
                release_camera_dmabuf_frame(active_cap, &camera_frame_info);
            }

            if (!frame_ok) continue;

            g_active_jobs.fetch_add(1);
            frame.copyTo(worker->ori_img);
            slot_states[i].busy.store(true);
            slot_states[i].ready.store(false);
            slot_states[i].thread_id = std::this_thread::get_id();
            
            // 保存原始图像尺寸（用于绘制检测框坐标转换）
            cv::Size img_size = frame.size();
            
            // 只在推理开启时才执行推理
            bool use_tracker = do_inference ? g_tracker_enabled.load() : false;
            int cam_id = (i < 3) ? 0 : 1;  // Slot 0-2 是摄像头0, Slot 3-5 是摄像头1
            int tracker_stream_id = g_video_mode.load() ? 2 : cam_id;
            try {
                std::thread([i, &slot_states, worker, do_inference, use_tracker, cam_id, tracker_stream_id, img_size]() {
                worker->set_tracker_stream_id(tracker_stream_id);
                worker->set_use_tracker(use_tracker);
                int ret = 0;
                if (do_inference) {
                    ret = worker->interf_detect_only();
                }
                if (ret == 0) {
                    // 更新图像尺寸（用于绘制）
                    g_current_img_size = img_size;

                    // 更新对应摄像头的检测数量
                    if (cam_id == 0) {
                        g_cam0_detection_count = rknn_lite::last_detection_count.load();
                    } else {
                        g_cam1_detection_count = rknn_lite::last_detection_count.load();
                    }

                    // 如果启用了跟踪，保存跟踪结果
                    if (use_tracker) {
                        std::lock_guard<std::mutex> lock(g_tracker_mutex);
                        auto tracks = worker->get_last_tracks();
                        if (i < (int)g_tracker_results.size()) {
                            g_tracker_results[i] = tracks;
                        }
                    }

                    // 仅在线程所有输出都写完后再标记 ready，避免主线程过早复用 slot
                    slot_states[i].ready.store(true);
                }
                // 不推理时，直接标记完成
                if (!do_inference) {
                    slot_states[i].ready.store(true);
                }
                slot_states[i].busy.store(false);
                g_active_jobs.fetch_sub(1);
                }).detach();
            } catch (...) {
                slot_states[i].busy.store(false);
                slot_states[i].ready.store(false);
                g_active_jobs.fetch_sub(1);
            }
        }
        
        if (!any_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // FPS 统计
        gettimeofday(&time_now, nullptr);
        long now_ms = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
        if (now_ms - last_fps_time_ms >= 1000) {
            float elapsed = (now_ms - last_fps_time_ms) / 1000.0f;
            g_cam0_fps.store((int)(frames0 / elapsed));
            g_cam1_fps.store((int)(frames1 / elapsed));
            
#ifdef USE_RTSP_MPP
            if (g_rtsp_streaming.load() && (push0 > 0 || push1 > 0)) {
                printf("[FPS] Cam0: %d FPS | Cam1: %d FPS | RTSP: %d | %d\n",
                       g_cam0_fps.load(), g_cam1_fps.load(), push0, push1);
            }
#else
            // 无 RTSP 时也打印 FPS
            if (frames0 > 0 || frames1 > 0) {
                printf("[FPS] Cam0: %d FPS | Cam1: %d FPS\n",
                       g_cam0_fps.load(), g_cam1_fps.load());
            }
#endif
            
            frames0 = 0;
            frames1 = 0;
            push0 = 0;
            push1 = 0;
            last_fps_time_ms = now_ms;
        }
    }
    
    printf("[Main] 等待退出...\n");
    http_thread.join();

    std::string unload_msg;
    if (g_model_loaded.load() || g_active_jobs.load() > 0) {
        if (!unload_model_runtime(&unload_msg)) {
            printf("[Main] 警告: %s\n", unload_msg.c_str());
        } else {
            printf("[Main] 模型资源已清理\n");
        }
    }

    if (cap0.isOpened()) cap0.release();
    if (cap1.isOpened()) cap1.release();
    v4l2_dmabuf::close_capture(&dmabuf_cap0);
    v4l2_dmabuf::close_capture(&dmabuf_cap1);

#ifdef USE_RTSP_MPP
    {
        std::lock_guard<std::mutex> lock(g_rtsp_mutex);
        g_video_rtsp_raw_stop = true;
        stop_rtsp_senders_locked();
    }
#endif

    // 清理视频资源
    stop_video_reader();

    printf("[Main] 服务已退出\n");
    return 0;
}
