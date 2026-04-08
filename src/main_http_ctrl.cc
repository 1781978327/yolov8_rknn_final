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
#include <netdb.h>
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
#include <cctype>
#include <cstdint>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <libdrm/drm_fourcc.h>
#ifdef USE_RTSP_MPP
#include <rockchip/mpp_buffer.h>
#endif

// RKNN 相关
#include "rknnPool.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "im2d.hpp"
#include "rga.h"
#include "http_ctrl_camera_io.h"
#include "http_ctrl_web_utils.h"
#include "v4l2_dmabuf_capture.h"

// RTSP 推流相关
#ifdef USE_RTSP_MPP
#include "http_ctrl_raw_video_rtsp.h"
#include "rtsp_mpp_sender.h"
#include "ffmpeg_rkmpp_reader.h"
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
std::atomic<int>  g_rtsp_cam0_fps(0), g_rtsp_cam1_fps(0), g_rtsp_video_fps(0);
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
cv::Mat g_latest_frame_cam[2];
std::mutex g_latest_frame_cam_mutex[2];
bool g_frame_available_cam[2] = {false, false};
int g_latest_frame_slot_cam[2] = {-1, -1};

// 跟踪器相关
std::atomic<bool> g_tracker_enabled(false);  // 0=关闭, 1=开启
std::mutex g_tracker_mutex;
std::vector<std::vector<TrackerResultItem>> g_tracker_results;  // 每个 slot 的跟踪结果

void draw_rtsp_fps_overlay(cv::Mat& frame, int fps_value);
std::string label_name_for_detection(const DetectionResultItem& det);

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
    g_rtsp_video_fps.store((int)(fps + 0.5f));
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
                if (!frame.empty()) {
                    draw_rtsp_fps_overlay(frame, g_rtsp_video_fps.load());
                    pushed_ok = g_rtsp_sender_video->push(frame);
                } else if (frame_info.valid) {
                    pushed_ok = g_rtsp_sender_video->push_dmabuf(
                        frame_info.fd,
                        frame_info.size,
                        frame_info.width,
                        frame_info.height,
                        frame_info.wstride,
                        frame_info.hstride,
                        frame_info.drm_format);
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
    g_rtsp_video_fps.store(0);
    g_video_rtsp_raw_running = false;
}

void start_video_rtsp_raw_thread_if_needed() {
    if (g_video_rtsp_raw_running.load()) {
        return;
    }
    // 线程结束后对象仍可能是 joinable，重新赋值前先回收，避免 std::terminate。
    if (g_video_rtsp_raw_thread.joinable()) {
        g_video_rtsp_raw_thread.join();
    }
    g_video_rtsp_raw_thread = std::thread(video_rtsp_raw_loop);
}
#endif

#ifdef USE_RTSP_MPP
struct SlotBgrDmabuf {
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    int size = 0;
    int width = 0;
    int height = 0;
    int wstride = 0;
    int hstride = 0;
    cv::Mat mat;
};

static void release_slot_bgr_dmabuf(SlotBgrDmabuf* slot) {
    if (!slot) return;
    slot->mat.release();
    if (slot->buffer) {
        mpp_buffer_put(slot->buffer);
    }
    *slot = SlotBgrDmabuf{};
}

static bool ensure_slot_bgr_dmabuf(SlotBgrDmabuf* slot, int width, int height) {
    if (!slot || width <= 0 || height <= 0) return false;
    if (slot->buffer && slot->ptr && slot->fd >= 0 &&
        slot->width == width && slot->height == height) {
        return true;
    }

    release_slot_bgr_dmabuf(slot);

    const int size = width * height * 3;
    MPP_RET ret = mpp_buffer_get(nullptr, &slot->buffer, size);
    if (ret != MPP_OK || !slot->buffer) {
        release_slot_bgr_dmabuf(slot);
        return false;
    }

    slot->fd = mpp_buffer_get_fd(slot->buffer);
    slot->ptr = mpp_buffer_get_ptr(slot->buffer);
    if (slot->fd < 0 || slot->ptr == nullptr) {
        release_slot_bgr_dmabuf(slot);
        return false;
    }

    slot->size = size;
    slot->width = width;
    slot->height = height;
    slot->wstride = width;
    slot->hstride = height;
    slot->mat = cv::Mat(height, width, CV_8UC3, slot->ptr, (size_t)width * 3);
    return !slot->mat.empty();
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
std::atomic<int> g_box_count_alert_threshold(0);  // 框数量告警阈值（0=关闭）
std::atomic<int> g_box_count_alert_cooldown_ms(8000);  // 告警冷却时间

std::mutex g_box_alert_state_mutex;
bool g_box_count_over_state[2] = {false, false};
long long g_last_box_alert_ms[2] = {0, 0};

std::string g_report_server_host = "127.0.0.1";
int g_report_server_port = 8080;
std::string g_report_server_path = "/api/detection/record/rknn/report";
int g_report_camera_id_cam0 = 1;
int g_report_camera_id_cam1 = 2;
std::string g_forbidden_area_path = "/api/rknn/forbidden-area";
std::atomic<int> g_forbidden_area_fetch_interval_ms(2000);
std::atomic<int> g_forbidden_area_fetch_timeout_ms(600);
std::atomic<int> g_intrusion_alert_cooldown_ms(8000);

struct ForbiddenAreaCache {
    bool valid = false;
    int image_width = 0;
    int image_height = 0;
    std::vector<cv::Point2f> points;
    long long last_fetch_ms = 0;
};

std::mutex g_forbidden_area_mutex;
ForbiddenAreaCache g_forbidden_area_cache[2];
bool g_forbidden_area_fetching[2] = {false, false};
std::mutex g_intrusion_alert_state_mutex;
bool g_intrusion_over_state[2] = {false, false};
long long g_last_intrusion_alert_ms[2] = {0, 0};

// 当前帧的图像尺寸（用于坐标转换）
cv::Size g_current_img_size(1280, 720);  // 默认值
int g_model_width = 640, g_model_height = 640;  // 模型输入尺寸（YOLOv8 默认 640）

// ---------------------- 工具函数 ----------------------
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

void draw_rtsp_fps_overlay(cv::Mat& frame, int fps_value) {
    if (frame.empty()) return;
    if (fps_value < 0) fps_value = 0;
    char text[64];
    snprintf(text, sizeof(text), "FPS: %d", fps_value);

    int baseline = 0;
    const double font_scale = 0.8;
    const int thickness = 2;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const int pad = 6;
    const int tx = 10;
    const int ty = 10 + text_size.height;
    int bx = std::max(0, tx - pad);
    int by = std::max(0, ty - text_size.height - pad);
    int bw = text_size.width + pad * 2;
    int bh = text_size.height + baseline + pad * 2;
    if (bx + bw > frame.cols) bw = std::max(1, frame.cols - bx);
    if (by + bh > frame.rows) bh = std::max(1, frame.rows - by);

    cv::rectangle(frame, cv::Rect(bx, by, bw, bh), cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, text, cv::Point(tx, ty),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness);
}

struct CpuTimes {
    uint64_t user = 0;
    uint64_t nice = 0;
    uint64_t system = 0;
    uint64_t idle = 0;
    uint64_t iowait = 0;
    uint64_t irq = 0;
    uint64_t softirq = 0;
    uint64_t steal = 0;
};

bool read_cpu_times(CpuTimes* out) {
    if (!out) return false;
    std::ifstream ifs("/proc/stat");
    if (!ifs.is_open()) return false;
    std::string line;
    if (!std::getline(ifs, line)) return false;
    std::istringstream iss(line);
    std::string cpu_tag;
    iss >> cpu_tag;
    if (cpu_tag != "cpu") return false;
    CpuTimes t;
    iss >> t.user >> t.nice >> t.system >> t.idle >> t.iowait >> t.irq >> t.softirq >> t.steal;
    *out = t;
    return true;
}

double calc_cpu_usage_percent(const CpuTimes& prev, const CpuTimes& cur) {
    const uint64_t idle_prev = prev.idle + prev.iowait;
    const uint64_t idle_cur = cur.idle + cur.iowait;
    const uint64_t non_idle_prev = prev.user + prev.nice + prev.system + prev.irq + prev.softirq + prev.steal;
    const uint64_t non_idle_cur = cur.user + cur.nice + cur.system + cur.irq + cur.softirq + cur.steal;
    const uint64_t total_prev = idle_prev + non_idle_prev;
    const uint64_t total_cur = idle_cur + non_idle_cur;
    if (total_cur <= total_prev) return -1.0;
    const double totald = (double)(total_cur - total_prev);
    const double idled = (double)(idle_cur - idle_prev);
    return std::max(0.0, std::min(100.0, (totald - idled) * 100.0 / totald));
}

double parse_npu_load_percent_text(const std::string& text) {
    std::vector<double> values;
    values.reserve(8);

    // 优先解析带 % 的格式，兼容如 "core0: 35%"
    for (size_t i = 0; i < text.size();) {
        if (!std::isdigit((unsigned char)text[i])) {
            ++i;
            continue;
        }
        size_t start = i;
        while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
        if (i < text.size() && text[i] == '.') {
            ++i;
            while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
        }
        if (i < text.size() && text[i] == '%') {
            values.push_back(strtod(text.substr(start, i - start).c_str(), nullptr));
            ++i;
        }
    }

    // 回退：不带 % 的纯数字输出（只取 0..100）
    if (values.empty()) {
        for (size_t i = 0; i < text.size();) {
            if (!std::isdigit((unsigned char)text[i])) {
                ++i;
                continue;
            }
            size_t start = i;
            while (i < text.size() && std::isdigit((unsigned char)text[i])) ++i;
            long v = strtol(text.substr(start, i - start).c_str(), nullptr, 10);
            if (v >= 0 && v <= 100) {
                values.push_back((double)v);
            }
        }
    }

    if (values.empty()) return -1.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / (double)values.size();
}

double read_npu_load_percent() {
    std::ifstream ifs("/sys/kernel/debug/rknpu/load");
    if (!ifs.is_open()) return -1.0;
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (content.empty()) return -1.0;
    return parse_npu_load_percent_text(content);
}

void usage_monitor_loop() {
    const auto sample_interval = std::chrono::seconds(1);
    const auto report_window = std::chrono::seconds(10);
    auto next_report = std::chrono::steady_clock::now() + report_window;

    CpuTimes prev_cpu;
    bool has_prev_cpu = read_cpu_times(&prev_cpu);
    bool npu_warned = false;
    std::vector<double> cpu_samples;
    std::vector<double> npu_samples;
    cpu_samples.reserve(16);
    npu_samples.reserve(16);
    long long cv_draw_total_us_window = 0;
    long long cv_draw_samples_window = 0;

    while (g_running.load()) {
        std::this_thread::sleep_for(sample_interval);
        if (!g_running.load()) break;

        CpuTimes cur_cpu;
        if (read_cpu_times(&cur_cpu)) {
            if (has_prev_cpu) {
                double cpu = calc_cpu_usage_percent(prev_cpu, cur_cpu);
                if (cpu >= 0.0) cpu_samples.push_back(cpu);
            }
            prev_cpu = cur_cpu;
            has_prev_cpu = true;
        }

        double npu = read_npu_load_percent();
        if (npu >= 0.0) {
            npu_samples.push_back(npu);
        } else if (!npu_warned) {
            printf("[Usage] 警告: 无法读取 NPU 负载(/sys/kernel/debug/rknpu/load)\n");
            npu_warned = true;
        }

        long long cv_sum_us = g_opencv_draw_total_us.exchange(0, std::memory_order_relaxed);
        long long cv_cnt = g_opencv_draw_sample_count.exchange(0, std::memory_order_relaxed);
        if (cv_sum_us > 0 && cv_cnt > 0) {
            cv_draw_total_us_window += cv_sum_us;
            cv_draw_samples_window += cv_cnt;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= next_report) {
            auto calc_avg = [](const std::vector<double>& samples) -> double {
                if (samples.empty()) return -1.0;
                double s = 0.0;
                for (double v : samples) s += v;
                return s / (double)samples.size();
            };
            double cpu_avg = calc_avg(cpu_samples);
            double npu_avg = calc_avg(npu_samples);
            double cv_draw_avg_ms = (cv_draw_samples_window > 0)
                                        ? (double)cv_draw_total_us_window / (double)cv_draw_samples_window / 1000.0
                                        : -1.0;
            auto metric_percent = [](double value) -> std::string {
                if (value < 0.0) return "N/A";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1f%%", value);
                return std::string(buf);
            };
            auto metric_ms = [](double value) -> std::string {
                if (value < 0.0) return "N/A";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2fms/帧", value);
                return std::string(buf);
            };
            std::string cpu_text = metric_percent(cpu_avg);
            std::string npu_text = metric_percent(npu_avg);
            std::string cv_text = metric_ms(cv_draw_avg_ms);
            printf("[Usage-10s] CPU平均: %s | NPU平均: %s | OpenCV绘制: %s\n",
                   cpu_text.c_str(), npu_text.c_str(), cv_text.c_str());
            cpu_samples.clear();
            npu_samples.clear();
            cv_draw_total_us_window = 0;
            cv_draw_samples_window = 0;
            next_report = now + report_window;
        }
    }
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

bool file_exists(const std::string& path) {
    return !path.empty() && access(path.c_str(), R_OK) == 0;
}

std::string parse_query_param(const std::string& path, const std::string& key) {
    size_t qpos = path.find('?');
    if (qpos == std::string::npos) return "";
    const std::string query = path.substr(qpos + 1);
    size_t start = 0;
    while (start <= query.size()) {
        size_t end = query.find('&', start);
        std::string part = query.substr(start, end == std::string::npos ? std::string::npos : (end - start));
        size_t eq = part.find('=');
        if (eq != std::string::npos) {
            std::string part_key = part.substr(0, eq);
            if (part_key == key) {
                std::string raw = part.substr(eq + 1);
                return url_decode(raw);
            }
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return "";
}

long long monotonic_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string local_time_iso8601() {
    std::time_t now = std::time(nullptr);
    std::tm tm_local;
    localtime_r(&now, &tm_local);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_local);
    return std::string(buf);
}

void update_latest_frame_global(const cv::Mat& frame, int slot) {
    if (frame.empty()) return;
    std::lock_guard<std::mutex> lock(g_latest_frame_mutex);
    g_latest_frame = frame.clone();
    g_frame_available = !g_latest_frame.empty();
    g_latest_frame_slot = slot;
}

void update_latest_frame_for_cam(int cam, const cv::Mat& frame, int slot) {
    if (cam < 0 || cam > 1) return;
    if (frame.empty()) return;
    std::lock_guard<std::mutex> lock(g_latest_frame_cam_mutex[cam]);
    g_latest_frame_cam[cam] = frame.clone();
    g_frame_available_cam[cam] = !g_latest_frame_cam[cam].empty();
    g_latest_frame_slot_cam[cam] = slot;
}

std::string base64_encode_bytes(const unsigned char* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = ((uint32_t)data[i]) << 16;
        bool has_b1 = (i + 1 < len);
        bool has_b2 = (i + 2 < len);
        if (has_b1) v |= ((uint32_t)data[i + 1]) << 8;
        if (has_b2) v |= (uint32_t)data[i + 2];

        out.push_back(table[(v >> 18) & 0x3F]);
        out.push_back(table[(v >> 12) & 0x3F]);
        out.push_back(has_b1 ? table[(v >> 6) & 0x3F] : '=');
        out.push_back(has_b2 ? table[v & 0x3F] : '=');
    }
    return out;
}

bool send_all_bytes(int sock, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(sock, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

int parse_http_status_code(const std::string& response) {
    size_t line_end = response.find("\r\n");
    std::string first_line = (line_end == std::string::npos) ? response : response.substr(0, line_end);
    size_t sp1 = first_line.find(' ');
    if (sp1 == std::string::npos) return -1;
    size_t sp2 = first_line.find(' ', sp1 + 1);
    std::string code_text = first_line.substr(sp1 + 1, (sp2 == std::string::npos) ? std::string::npos : (sp2 - sp1 - 1));
    char* endptr = nullptr;
    long code = strtol(code_text.c_str(), &endptr, 10);
    if (endptr == code_text.c_str() || *endptr != '\0') return -1;
    return (int)code;
}

bool http_request_to_report_server(
    const std::string& method,
    const std::string& request_path,
    const std::string& content_type,
    const std::string& body,
    int timeout_ms,
    std::string* response_head,
    std::string* response_body,
    int* status_code_out) {

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }

    if (timeout_ms < 100) timeout_ms = 100;
    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons((uint16_t)g_report_server_port);

    if (inet_pton(AF_INET, g_report_server_host.c_str(), &serv_addr.sin_addr) <= 0) {
        struct hostent* server = gethostbyname(g_report_server_host.c_str());
        if (server == nullptr || server->h_length <= 0) {
            close(sock);
            return false;
        }
        memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, (size_t)server->h_length);
    }

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sock);
        return false;
    }

    std::string path = request_path;
    if (path.empty()) path = "/";
    if (path[0] != '/') {
        path = "/" + path;
    }

    std::ostringstream oss;
    oss << method << " " << path << " HTTP/1.1\r\n";
    oss << "Host: " << g_report_server_host << ":" << g_report_server_port << "\r\n";
    if (!content_type.empty()) {
        oss << "Content-Type: " << content_type << "\r\n";
    }
    if (!body.empty()) {
        oss << "Content-Length: " << body.size() << "\r\n";
    }
    oss << "Connection: close\r\n\r\n";
    if (!body.empty()) {
        oss << body;
    }
    const std::string req = oss.str();

    bool sent_ok = send_all_bytes(sock, req.data(), req.size());
    if (!sent_ok) {
        close(sock);
        return false;
    }

    std::string response;
    char resp_buf[2048];
    while (true) {
        int n = recv(sock, resp_buf, sizeof(resp_buf), 0);
        if (n <= 0) break;
        response.append(resp_buf, (size_t)n);
    }
    close(sock);

    if (response.empty()) return false;
    if (response_head) {
        size_t head_end = response.find("\r\n\r\n");
        *response_head = response.substr(0, head_end == std::string::npos ? std::min((size_t)256, response.size()) : head_end);
    }
    if (response_body) {
        size_t body_pos = response.find("\r\n\r\n");
        *response_body = (body_pos == std::string::npos) ? "" : response.substr(body_pos + 4);
    }

    int status_code = parse_http_status_code(response);
    if (status_code_out) {
        *status_code_out = status_code;
    }
    return status_code >= 200 && status_code < 300;
}

bool post_json_to_report_server(const std::string& body, std::string* response_head) {
    int status_code = -1;
    return http_request_to_report_server(
        "POST",
        g_report_server_path,
        "application/json",
        body,
        3000,
        response_head,
        nullptr,
        &status_code);
}

void report_detection_count_alert_worker(int cam_id, int detection_count, int threshold, cv::Mat frame_bgr) {
    if (frame_bgr.empty()) return;

    std::vector<uchar> jpg_buf;
    if (!cv::imencode(".jpg", frame_bgr, jpg_buf, {cv::IMWRITE_JPEG_QUALITY, 85})) {
        printf("[AlertReport] JPEG 编码失败: cam=%d count=%d\n", cam_id, detection_count);
        return;
    }

    std::string image_b64 = base64_encode_bytes(jpg_buf.data(), jpg_buf.size());
    int camera_id = (cam_id == 0) ? g_report_camera_id_cam0 : g_report_camera_id_cam1;

    std::ostringstream detection_result;
    detection_result << "{\"type\":\"box_count_exceeded\",\"cam\":" << cam_id
                     << ",\"count\":" << detection_count
                     << ",\"threshold\":" << threshold << "}";

    std::ostringstream body;
    body << "{";
    body << "\"cameraId\":" << camera_id << ",";
    body << "\"detectionTime\":\"" << local_time_iso8601() << "\",";
    body << "\"detectionResult\":\"" << json_escape(detection_result.str()) << "\",";
    body << "\"imageBase64\":\"" << image_b64 << "\"";
    body << "}";

    std::string response_head;
    bool ok = post_json_to_report_server(body.str(), &response_head);
    if (ok) {
        printf("[AlertReport] 上报成功: cam=%d cameraId=%d count=%d threshold=%d\n",
               cam_id, camera_id, detection_count, threshold);
    } else {
        std::string brief = response_head.empty() ? "(no response)" : response_head.substr(0, 120);
        printf("[AlertReport] 上报失败: cam=%d cameraId=%d count=%d threshold=%d resp=%s\n",
               cam_id, camera_id, detection_count, threshold, brief.c_str());
    }
}

void report_detection_count_alert_async(int cam_id, int detection_count, int threshold, const cv::Mat& frame_bgr) {
    if (frame_bgr.empty()) return;
    cv::Mat frame_copy = frame_bgr.clone();
    if (frame_copy.empty()) return;
    try {
        std::thread(report_detection_count_alert_worker, cam_id, detection_count, threshold, frame_copy).detach();
    } catch (...) {
        printf("[AlertReport] 启动上报线程失败: cam=%d count=%d\n", cam_id, detection_count);
    }
}

void maybe_trigger_detection_count_alert(int cam_id, int detection_count, const cv::Mat& frame_bgr) {
    if (cam_id < 0 || cam_id > 1) return;

    int threshold = g_box_count_alert_threshold.load();
    if (threshold <= 0) {
        std::lock_guard<std::mutex> lock(g_box_alert_state_mutex);
        g_box_count_over_state[cam_id] = false;
        return;
    }

    bool should_report = false;
    {
        std::lock_guard<std::mutex> lock(g_box_alert_state_mutex);
        bool over = detection_count > threshold;
        bool was_over = g_box_count_over_state[cam_id];
        g_box_count_over_state[cam_id] = over;
        if (!over || was_over) {
            return;
        }

        long long now_ms = monotonic_now_ms();
        int cooldown_ms = g_box_count_alert_cooldown_ms.load();
        if (cooldown_ms < 0) cooldown_ms = 0;
        if (now_ms - g_last_box_alert_ms[cam_id] < cooldown_ms) {
            return;
        }
        g_last_box_alert_ms[cam_id] = now_ms;
        should_report = true;
    }

    if (should_report) {
        report_detection_count_alert_async(cam_id, detection_count, threshold, frame_bgr);
    }
}

bool json_extract_int_after_key(const std::string& text, size_t key_pos, int* out) {
    if (!out || key_pos == std::string::npos) return false;
    size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
    bool neg = false;
    if (i < text.size() && text[i] == '-') {
        neg = true;
        ++i;
    }
    if (i >= text.size() || !std::isdigit((unsigned char)text[i])) return false;
    long value = 0;
    while (i < text.size() && std::isdigit((unsigned char)text[i])) {
        value = value * 10 + (text[i] - '0');
        ++i;
    }
    if (neg) value = -value;
    *out = (int)value;
    return true;
}

bool json_extract_int_field(const std::string& text, const std::string& key, int* out) {
    std::string token = "\"" + key + "\"";
    size_t key_pos = text.find(token);
    if (key_pos == std::string::npos) return false;
    return json_extract_int_after_key(text, key_pos, out);
}

bool json_extract_bool_field(const std::string& text, const std::string& key, bool* out) {
    if (!out) return false;
    std::string token = "\"" + key + "\"";
    size_t key_pos = text.find(token);
    if (key_pos == std::string::npos) return false;
    size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
    if (text.compare(i, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (text.compare(i, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool parse_points_from_forbidden_area_body(const std::string& body, std::vector<cv::Point2f>* points_out) {
    if (!points_out) return false;
    points_out->clear();

    size_t key_pos = body.find("\"points\"");
    if (key_pos == std::string::npos) return false;
    size_t arr_begin = body.find('[', key_pos);
    if (arr_begin == std::string::npos) return false;

    size_t depth = 0;
    size_t arr_end = std::string::npos;
    for (size_t i = arr_begin; i < body.size(); ++i) {
        if (body[i] == '[') ++depth;
        else if (body[i] == ']') {
            if (depth == 0) return false;
            --depth;
            if (depth == 0) {
                arr_end = i;
                break;
            }
        }
    }
    if (arr_end == std::string::npos) return false;

    std::string arr_text = body.substr(arr_begin + 1, arr_end - arr_begin - 1);
    size_t cursor = 0;
    while (cursor < arr_text.size()) {
        size_t x_pos = arr_text.find("\"x\"", cursor);
        if (x_pos == std::string::npos) break;
        int x = 0;
        if (!json_extract_int_after_key(arr_text, x_pos, &x)) {
            cursor = x_pos + 3;
            continue;
        }
        size_t y_pos = arr_text.find("\"y\"", x_pos);
        if (y_pos == std::string::npos) break;
        int y = 0;
        if (!json_extract_int_after_key(arr_text, y_pos, &y)) {
            cursor = y_pos + 3;
            continue;
        }
        points_out->push_back(cv::Point2f((float)x, (float)y));
        cursor = y_pos + 3;
    }
    return !points_out->empty();
}

int forbidden_area_camera_id_from_cam(int cam_id) {
    return (cam_id == 0) ? 1 : 2;
}

bool fetch_forbidden_area_from_server(int cam_id, ForbiddenAreaCache* cache_out, std::string* err_msg) {
    if (!cache_out) return false;
    *cache_out = ForbiddenAreaCache{};
    cache_out->last_fetch_ms = monotonic_now_ms();

    std::string path = g_forbidden_area_path;
    if (path.empty()) path = "/api/rknn/forbidden-area";
    if (path[0] != '/') path = "/" + path;
    path += (path.find('?') == std::string::npos) ? "?" : "&";
    path += "cameraId=" + std::to_string(forbidden_area_camera_id_from_cam(cam_id));

    std::string response_head;
    std::string response_body;
    int status_code = -1;
    int timeout_ms = g_forbidden_area_fetch_timeout_ms.load();
    bool ok = http_request_to_report_server("GET", path, "application/json", "",
                                            timeout_ms, &response_head, &response_body, &status_code);
    if (!ok) {
        if (err_msg) {
            std::ostringstream oss;
            oss << "HTTP fail status=" << status_code << " head="
                << (response_head.empty() ? "(empty)" : response_head.substr(0, 120));
            *err_msg = oss.str();
        }
        return false;
    }

    bool exists = false;
    if (!json_extract_bool_field(response_body, "exists", &exists)) {
        // 未匹配到 exists，按“无有效区域”处理，避免解析异常时阻塞主流程
        cache_out->valid = false;
        return true;
    }
    if (!exists) {
        cache_out->valid = false;
        return true;
    }

    int img_w = 0;
    int img_h = 0;
    (void)json_extract_int_field(response_body, "imageWidth", &img_w);
    (void)json_extract_int_field(response_body, "imageHeight", &img_h);

    std::vector<cv::Point2f> points;
    if (!parse_points_from_forbidden_area_body(response_body, &points) || points.size() < 4) {
        cache_out->valid = false;
        if (err_msg) *err_msg = "points parse failed or less than 4";
        return true;
    }

    if (points.size() > 4) {
        points.resize(4);
    }
    cache_out->valid = true;
    cache_out->image_width = img_w;
    cache_out->image_height = img_h;
    cache_out->points = points;
    return true;
}

void refresh_forbidden_area_cache_if_needed(int cam_id) {
    if (cam_id < 0 || cam_id > 1) return;

    long long now_ms = monotonic_now_ms();
    int interval_ms = g_forbidden_area_fetch_interval_ms.load();
    if (interval_ms < 500) interval_ms = 500;

    {
        std::lock_guard<std::mutex> lock(g_forbidden_area_mutex);
        if (g_forbidden_area_fetching[cam_id]) return;
        if (now_ms - g_forbidden_area_cache[cam_id].last_fetch_ms < interval_ms) return;
        g_forbidden_area_fetching[cam_id] = true;
    }

    ForbiddenAreaCache fetched;
    std::string err_msg;
    bool fetch_ok = fetch_forbidden_area_from_server(cam_id, &fetched, &err_msg);

    static std::atomic<int> fetch_fail_counter[2];
    {
        std::lock_guard<std::mutex> lock(g_forbidden_area_mutex);
        g_forbidden_area_fetching[cam_id] = false;
        if (fetch_ok) {
            g_forbidden_area_cache[cam_id] = fetched;
            fetch_fail_counter[cam_id].store(0);
        } else {
            g_forbidden_area_cache[cam_id].last_fetch_ms = now_ms;
            int fail_n = fetch_fail_counter[cam_id].fetch_add(1) + 1;
            if (fail_n <= 3 || (fail_n % 60) == 0) {
                printf("[ForbiddenArea] 拉取失败: cam=%d err=%s\n", cam_id, err_msg.c_str());
            }
        }
    }
}

bool get_forbidden_area_polygon_for_frame(int cam_id, const cv::Size& frame_size, std::vector<cv::Point2f>* polygon_out) {
    if (!polygon_out) return false;
    polygon_out->clear();
    if (cam_id < 0 || cam_id > 1) return false;
    if (frame_size.width <= 0 || frame_size.height <= 0) return false;

    ForbiddenAreaCache cache;
    {
        std::lock_guard<std::mutex> lock(g_forbidden_area_mutex);
        cache = g_forbidden_area_cache[cam_id];
    }
    if (!cache.valid || cache.points.size() != 4) return false;

    float sx = 1.0f;
    float sy = 1.0f;
    if (cache.image_width > 0) sx = (float)frame_size.width / (float)cache.image_width;
    if (cache.image_height > 0) sy = (float)frame_size.height / (float)cache.image_height;

    polygon_out->reserve(cache.points.size());
    for (const auto& p : cache.points) {
        float x = p.x * sx;
        float y = p.y * sy;
        if (x < 0.0f) x = 0.0f;
        if (y < 0.0f) y = 0.0f;
        if (x > (float)(frame_size.width - 1)) x = (float)(frame_size.width - 1);
        if (y > (float)(frame_size.height - 1)) y = (float)(frame_size.height - 1);
        polygon_out->push_back(cv::Point2f(x, y));
    }
    return polygon_out->size() == 4;
}

bool collect_intrusion_hits_for_frame(
    int cam_id,
    const cv::Size& frame_size,
    const std::vector<DetectionResultItem>& detections,
    std::vector<cv::Point2f>* polygon_out,
    std::vector<DetectionResultItem>* hit_detections_out) {

    if (!polygon_out || !hit_detections_out) return false;
    polygon_out->clear();
    hit_detections_out->clear();
    if (cam_id < 0 || cam_id > 1) return false;
    if (frame_size.width <= 0 || frame_size.height <= 0) return false;

    if (!get_forbidden_area_polygon_for_frame(cam_id, frame_size, polygon_out)) {
        return false;
    }

    for (const auto& det : detections) {
        float cx = (det.x1 + det.x2) * 0.5f;
        float cy = (det.y1 + det.y2) * 0.5f;
        if (cv::pointPolygonTest(*polygon_out, cv::Point2f(cx, cy), false) >= 0.0) {
            hit_detections_out->push_back(det);
        }
    }
    return true;
}

void draw_forbidden_area_overlay_if_available(int cam_id, cv::Mat& frame_bgr) {
    if (frame_bgr.empty()) return;
    refresh_forbidden_area_cache_if_needed(cam_id);
    std::vector<cv::Point2f> polygon;
    if (!get_forbidden_area_polygon_for_frame(cam_id, frame_bgr.size(), &polygon)) return;

    std::vector<cv::Point> poly_i;
    poly_i.reserve(polygon.size());
    for (const auto& p : polygon) {
        poly_i.push_back(cv::Point((int)(p.x + 0.5f), (int)(p.y + 0.5f)));
    }
    if (poly_i.size() < 3) return;

    const cv::Point* pts = poly_i.data();
    int npts = (int)poly_i.size();
    cv::polylines(frame_bgr, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 2);
    for (const auto& p : poly_i) {
        cv::circle(frame_bgr, p, 3, cv::Scalar(0, 0, 255), -1);
    }
}

void draw_intrusion_boxes_overlay_if_needed(
    int cam_id,
    cv::Mat& frame_bgr,
    const std::vector<DetectionResultItem>& detections) {

    if (frame_bgr.empty()) return;
    if (detections.empty()) return;

    std::vector<cv::Point2f> polygon;
    std::vector<DetectionResultItem> hits;
    if (!collect_intrusion_hits_for_frame(cam_id, frame_bgr.size(), detections, &polygon, &hits)) return;
    if (hits.empty()) return;

    for (const auto& det : hits) {
        int x1 = std::max(0, std::min((int)det.x1, frame_bgr.cols - 1));
        int y1 = std::max(0, std::min((int)det.y1, frame_bgr.rows - 1));
        int x2 = std::max(0, std::min((int)det.x2, frame_bgr.cols - 1));
        int y2 = std::max(0, std::min((int)det.y2, frame_bgr.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        cv::rectangle(frame_bgr, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);

        std::string label = label_name_for_detection(det);
        char text[192];
        snprintf(text, sizeof(text), "INTR %s %.1f%%", label.c_str(), det.score * 100.0f);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int tx = x1;
        int ty = y1 - ts.height - baseline;
        if (ty < 0) ty = 0;
        if (tx + ts.width > frame_bgr.cols) tx = std::max(0, frame_bgr.cols - ts.width);
        cv::rectangle(frame_bgr, cv::Rect(tx, ty, ts.width, ts.height + baseline), cv::Scalar(0, 0, 255), -1);
        cv::putText(frame_bgr, text, cv::Point(tx, ty + ts.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

std::vector<DetectionResultItem> build_detections_from_tracks(const std::vector<TrackerResultItem>& tracks) {
    std::vector<DetectionResultItem> detections;
    detections.reserve(tracks.size());
    for (const auto& t : tracks) {
        if (!t.active) continue;
        DetectionResultItem d;
        d.label = t.label;
        d.score = t.score;
        d.x1 = t.x1;
        d.y1 = t.y1;
        d.x2 = t.x2;
        d.y2 = t.y2;
        detections.push_back(d);
    }
    return detections;
}

std::string label_name_for_detection(const DetectionResultItem& det) {
    if (det.label >= 0 && det.label < (int)coco_labels.size()) {
        return coco_labels[det.label];
    }
    return "unknown";
}

void report_forbidden_area_intrusion_alert_worker(
    int cam_id,
    std::vector<DetectionResultItem> hit_detections,
    std::vector<cv::Point2f> polygon,
    cv::Mat frame_bgr) {

    if (frame_bgr.empty()) return;
    if (hit_detections.empty()) return;

    std::vector<cv::Point> poly_i;
    poly_i.reserve(polygon.size());
    for (const auto& p : polygon) {
        poly_i.push_back(cv::Point((int)(p.x + 0.5f), (int)(p.y + 0.5f)));
    }
    if (poly_i.size() >= 3) {
        const cv::Point* pts = poly_i.data();
        int npts = (int)poly_i.size();
        cv::polylines(frame_bgr, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 3);
    }
    for (const auto& p : poly_i) {
        cv::circle(frame_bgr, p, 4, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto& det : hit_detections) {
        int x1 = std::max(0, std::min((int)det.x1, frame_bgr.cols - 1));
        int y1 = std::max(0, std::min((int)det.y1, frame_bgr.rows - 1));
        int x2 = std::max(0, std::min((int)det.x2, frame_bgr.cols - 1));
        int y2 = std::max(0, std::min((int)det.y2, frame_bgr.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        cv::rectangle(frame_bgr, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
    }

    std::vector<uchar> jpg_buf;
    if (!cv::imencode(".jpg", frame_bgr, jpg_buf, {cv::IMWRITE_JPEG_QUALITY, 85})) {
        printf("[IntrusionReport] JPEG 编码失败: cam=%d\n", cam_id);
        return;
    }
    std::string image_b64 = base64_encode_bytes(jpg_buf.data(), jpg_buf.size());
    int camera_id = (cam_id == 0) ? g_report_camera_id_cam0 : g_report_camera_id_cam1;

    std::ostringstream detection_result;
    detection_result << "{\"type\":\"env_intrusion\",\"cam\":" << cam_id
                     << ",\"hitCount\":" << hit_detections.size()
                     << ",\"zonePoints\":[";
    for (size_t i = 0; i < polygon.size(); ++i) {
        if (i > 0) detection_result << ",";
        detection_result << "{\"x\":" << (int)(polygon[i].x + 0.5f)
                         << ",\"y\":" << (int)(polygon[i].y + 0.5f) << "}";
    }
    detection_result << "],\"objects\":[";
    for (size_t i = 0; i < hit_detections.size(); ++i) {
        const auto& det = hit_detections[i];
        if (i > 0) detection_result << ",";
        float cx = (det.x1 + det.x2) * 0.5f;
        float cy = (det.y1 + det.y2) * 0.5f;
        detection_result << "{\"label\":\"" << json_escape(label_name_for_detection(det)) << "\""
                         << ",\"cls\":" << det.label
                         << ",\"score\":" << std::fixed << std::setprecision(3) << det.score
                         << ",\"center\":{\"x\":" << (int)(cx + 0.5f) << ",\"y\":" << (int)(cy + 0.5f) << "}"
                         << ",\"box\":{\"x1\":" << (int)(det.x1 + 0.5f)
                         << ",\"y1\":" << (int)(det.y1 + 0.5f)
                         << ",\"x2\":" << (int)(det.x2 + 0.5f)
                         << ",\"y2\":" << (int)(det.y2 + 0.5f) << "}}";
    }
    detection_result << "]}";

    std::ostringstream body;
    body << "{";
    body << "\"cameraId\":" << camera_id << ",";
    body << "\"detectionTime\":\"" << local_time_iso8601() << "\",";
    body << "\"detectionResult\":\"" << json_escape(detection_result.str()) << "\",";
    body << "\"imageBase64\":\"" << image_b64 << "\"";
    body << "}";

    std::string response_head;
    bool ok = post_json_to_report_server(body.str(), &response_head);
    if (ok) {
        printf("[IntrusionReport] 上报成功: cam=%d cameraId=%d hit=%zu\n",
               cam_id, camera_id, hit_detections.size());
    } else {
        std::string brief = response_head.empty() ? "(no response)" : response_head.substr(0, 120);
        printf("[IntrusionReport] 上报失败: cam=%d cameraId=%d hit=%zu resp=%s\n",
               cam_id, camera_id, hit_detections.size(), brief.c_str());
    }
}

void report_forbidden_area_intrusion_alert_async(
    int cam_id,
    const std::vector<DetectionResultItem>& hit_detections,
    const std::vector<cv::Point2f>& polygon,
    const cv::Mat& frame_bgr) {

    if (frame_bgr.empty() || hit_detections.empty()) return;
    cv::Mat frame_copy = frame_bgr.clone();
    if (frame_copy.empty()) return;
    try {
        std::thread(report_forbidden_area_intrusion_alert_worker, cam_id, hit_detections, polygon, frame_copy).detach();
    } catch (...) {
        printf("[IntrusionReport] 启动上报线程失败: cam=%d\n", cam_id);
    }
}

void maybe_trigger_forbidden_area_intrusion_alert(
    int cam_id,
    const std::vector<DetectionResultItem>& detections,
    const cv::Mat& frame_bgr) {

    if (cam_id < 0 || cam_id > 1) return;
    if (frame_bgr.empty()) return;

    refresh_forbidden_area_cache_if_needed(cam_id);

    std::vector<cv::Point2f> polygon;
    std::vector<DetectionResultItem> hit_detections;
    if (!collect_intrusion_hits_for_frame(cam_id, frame_bgr.size(), detections, &polygon, &hit_detections)) {
        std::lock_guard<std::mutex> lock(g_intrusion_alert_state_mutex);
        g_intrusion_over_state[cam_id] = false;
        return;
    }

    bool intrusion = !hit_detections.empty();
    bool should_report = false;
    {
        std::lock_guard<std::mutex> lock(g_intrusion_alert_state_mutex);
        bool was_intrusion = g_intrusion_over_state[cam_id];
        g_intrusion_over_state[cam_id] = intrusion;
        if (!intrusion || was_intrusion) {
            return;
        }

        long long now_ms = monotonic_now_ms();
        int cooldown_ms = g_intrusion_alert_cooldown_ms.load();
        if (cooldown_ms < 0) cooldown_ms = 0;
        if (now_ms - g_last_intrusion_alert_ms[cam_id] < cooldown_ms) {
            return;
        }
        g_last_intrusion_alert_ms[cam_id] = now_ms;
        should_report = true;
    }

    if (should_report) {
        report_forbidden_area_intrusion_alert_async(cam_id, hit_detections, polygon, frame_bgr);
    }
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

        cv::Scalar color = tracker_color_for_item(track.label, track.track_id);
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
        const int draw_trail_points = get_track_draw_points_limit();
        int start_idx = (traj.size() > (size_t)draw_trail_points) ? ((int)traj.size() - draw_trail_points) : 0;
        for (size_t j = (size_t)start_idx + 1; j < traj.size(); j++) {
            int px1 = std::max(0, std::min((int)traj[j - 1].first, img.cols - 1));
            int py1 = std::max(0, std::min((int)traj[j - 1].second, img.rows - 1));
            int px2 = std::max(0, std::min((int)traj[j].first, img.cols - 1));
            int py2 = std::max(0, std::min((int)traj[j].second, img.rows - 1));
            cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2), color, 1);
        }
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
        bool forbidden_cam0_loaded = false;
        bool forbidden_cam1_loaded = false;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            model_path = g_model_path;
            label_path = g_label_path;
            model_error = g_model_error;
        }
        {
            std::lock_guard<std::mutex> lock(g_forbidden_area_mutex);
            forbidden_cam0_loaded = g_forbidden_area_cache[0].valid;
            forbidden_cam1_loaded = g_forbidden_area_cache[1].valid;
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
        oss << "\"box_count_alert_threshold\":" << g_box_count_alert_threshold.load() << ",";
        oss << "\"box_count_alert_cooldown_ms\":" << g_box_count_alert_cooldown_ms.load() << ",";
        oss << "\"forbidden_area_path\":\"" << json_escape(g_forbidden_area_path) << "\",";
        oss << "\"forbidden_area_fetch_interval_ms\":" << g_forbidden_area_fetch_interval_ms.load() << ",";
        oss << "\"forbidden_area_fetch_timeout_ms\":" << g_forbidden_area_fetch_timeout_ms.load() << ",";
        oss << "\"intrusion_alert_cooldown_ms\":" << g_intrusion_alert_cooldown_ms.load() << ",";
        oss << "\"forbidden_area_cam0_loaded\":" << (forbidden_cam0_loaded ? "true" : "false") << ",";
        oss << "\"forbidden_area_cam1_loaded\":" << (forbidden_cam1_loaded ? "true" : "false") << ",";
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
        // 设置检测阈值: /api/threshold/set?value=0.6&box_count=3
        float value = rknn_lite::get_detection_threshold();
        std::string value_text = parse_query_param(path, "value");
        if (!value_text.empty()) {
            try {
                value = std::stof(value_text);
            } catch (...) {
                send_response(client_fd,
                    build_json_response("error", "无效的 value 参数，必须是 0~1 浮点数"),
                    "application/json", 400);
                return;
            }
        }
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        g_confidence_threshold.store(value);
        rknn_lite::set_detection_threshold(value);

        bool has_box_count_param =
            (path.find("box_count=") != std::string::npos) ||
            (path.find("boxCount=") != std::string::npos);
        if (has_box_count_param) {
            std::string box_count_text = parse_query_param(path, "box_count");
            if (box_count_text.empty()) {
                box_count_text = parse_query_param(path, "boxCount");
            }
            char* endptr = nullptr;
            long parsed = strtol(box_count_text.c_str(), &endptr, 10);
            if (box_count_text.empty() || endptr == box_count_text.c_str() || *endptr != '\0' || parsed < 0) {
                send_response(client_fd,
                    build_json_response("error", "无效的 box_count 参数，必须是大于等于 0 的整数"),
                    "application/json", 400);
                return;
            }
            if (parsed > 10000) parsed = 10000;
            g_box_count_alert_threshold.store((int)parsed);
        }

        std::ostringstream msg;
        msg << "阈值已设置为: " << value;
        if (has_box_count_param) {
            msg << ", 框数量阈值: " << g_box_count_alert_threshold.load();
        }
        send_response(client_fd, build_json_response("success", msg.str()), "application/json");
    }
    else if (route == "/api/threshold/get" && method == "GET") {
        // 获取当前阈值（置信度 + 框数量告警阈值）
        std::ostringstream oss;
        oss << "{";
        oss << "\"threshold\":" << rknn_lite::get_detection_threshold() << ",";
        oss << "\"box_count_threshold\":" << g_box_count_alert_threshold.load();
        oss << "}";
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
        // 解析参数: /api/frame?track=1&redraw=1&cam=0
        bool want_tracker = g_tracker_enabled.load();
        if (path.find("track=1") != std::string::npos || path.find("track=ON") != std::string::npos) {
            want_tracker = true;
        } else if (path.find("track=0") != std::string::npos || path.find("track=OFF") != std::string::npos) {
            want_tracker = false;
        }
        // 默认不二次绘制：推理线程已经在 ori_img 上画过框，二次叠加会出现“一人双框”
        bool force_redraw = (path.find("redraw=1") != std::string::npos || path.find("overlay=1") != std::string::npos);
        int requested_cam = -1;
        std::string cam_text = parse_query_param(path, "cam");
        if (!cam_text.empty()) {
            char* endptr = nullptr;
            long parsed = strtol(cam_text.c_str(), &endptr, 10);
            if (endptr != cam_text.c_str() && *endptr == '\0' && parsed >= 0 && parsed <= 1) {
                requested_cam = (int)parsed;
            }
        }
        if (requested_cam < 0) {
            std::string camera_id_text = parse_query_param(path, "cameraId");
            if (!camera_id_text.empty()) {
                char* endptr = nullptr;
                long camera_id = strtol(camera_id_text.c_str(), &endptr, 10);
                if (endptr != camera_id_text.c_str() && *endptr == '\0') {
                    if (camera_id == 1) requested_cam = 0;
                    else if (camera_id == 2) requested_cam = 1;
                }
            }
        }
        
        cv::Mat frame_copy;
        bool available = false;
        int frame_slot = -1;
        if (requested_cam >= 0 && requested_cam <= 1) {
            std::lock_guard<std::mutex> lock(g_latest_frame_cam_mutex[requested_cam]);
            if (!g_latest_frame_cam[requested_cam].empty()) {
                frame_copy = g_latest_frame_cam[requested_cam].clone();
                available = true;
                frame_slot = g_latest_frame_slot_cam[requested_cam];
            }
        }
        if (!available) {
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
                if ((skip_redraw_log_counter++ % 600) == 0) {
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
    const char* env_rtsp_prebind_dma = getenv("RTSP_PREBIND_DMA");
    const char* env_tracker_backend = getenv("TRACKER_BACKEND");
    const char* env_tracker_reid = getenv("TRACKER_REID_MODEL");
    const char* env_report_host = getenv("REPORT_SERVER_HOST");
    const char* env_report_port = getenv("REPORT_SERVER_PORT");
    const char* env_report_path = getenv("REPORT_SERVER_PATH");
    const char* env_report_cam0_id = getenv("REPORT_CAM0_ID");
    const char* env_report_cam1_id = getenv("REPORT_CAM1_ID");
    const char* env_box_alert_cooldown = getenv("BOX_ALERT_COOLDOWN_MS");
    const char* env_forbidden_area_path = getenv("FORBIDDEN_AREA_PATH");
    const char* env_forbidden_area_sync_ms = getenv("FORBIDDEN_AREA_SYNC_MS");
    const char* env_forbidden_area_timeout_ms = getenv("FORBIDDEN_AREA_TIMEOUT_MS");
    const char* env_intrusion_alert_cooldown = getenv("INTRUSION_ALERT_COOLDOWN_MS");
    bool rtsp_prebind_dma = false;
    if (env_rtsp_host && *env_rtsp_host) {
        g_rtsp_host = env_rtsp_host;
    }
    if (env_rtsp_port && *env_rtsp_port) {
        int p = atoi(env_rtsp_port);
        if (p > 0 && p <= 65535) g_rtsp_port = p;
    }
    if (env_rtsp_prebind_dma && *env_rtsp_prebind_dma) {
        std::string v = env_rtsp_prebind_dma;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        rtsp_prebind_dma = (v == "1" || v == "true" || v == "on" || v == "yes");
    }
    if (env_tracker_backend && *env_tracker_backend) {
        if (!rknn_lite::set_tracker_backend(env_tracker_backend)) {
            printf("[Tracker] 警告: 无效的 TRACKER_BACKEND=%s，继续使用默认 bytetrack\n", env_tracker_backend);
        }
    }
    if (env_tracker_reid && *env_tracker_reid) {
        rknn_lite::set_tracker_reid_model_override(env_tracker_reid);
    }
    if (env_report_host && *env_report_host) {
        g_report_server_host = env_report_host;
    }
    if (env_report_port && *env_report_port) {
        int p = atoi(env_report_port);
        if (p > 0 && p <= 65535) g_report_server_port = p;
    }
    if (env_report_path && *env_report_path) {
        g_report_server_path = env_report_path;
        if (!g_report_server_path.empty() && g_report_server_path[0] != '/') {
            g_report_server_path = "/" + g_report_server_path;
        }
    }
    if (env_report_cam0_id && *env_report_cam0_id) {
        g_report_camera_id_cam0 = atoi(env_report_cam0_id);
    }
    if (env_report_cam1_id && *env_report_cam1_id) {
        g_report_camera_id_cam1 = atoi(env_report_cam1_id);
    }
    if (env_box_alert_cooldown && *env_box_alert_cooldown) {
        int cooldown = atoi(env_box_alert_cooldown);
        if (cooldown >= 0) g_box_count_alert_cooldown_ms.store(cooldown);
    }
    if (env_forbidden_area_path && *env_forbidden_area_path) {
        g_forbidden_area_path = env_forbidden_area_path;
        if (!g_forbidden_area_path.empty() && g_forbidden_area_path[0] != '/') {
            g_forbidden_area_path = "/" + g_forbidden_area_path;
        }
    }
    if (env_forbidden_area_sync_ms && *env_forbidden_area_sync_ms) {
        int v = atoi(env_forbidden_area_sync_ms);
        if (v > 0) g_forbidden_area_fetch_interval_ms.store(v);
    }
    if (env_forbidden_area_timeout_ms && *env_forbidden_area_timeout_ms) {
        int v = atoi(env_forbidden_area_timeout_ms);
        if (v > 0) g_forbidden_area_fetch_timeout_ms.store(v);
    }
    if (env_intrusion_alert_cooldown && *env_intrusion_alert_cooldown) {
        int v = atoi(env_intrusion_alert_cooldown);
        if (v >= 0) g_intrusion_alert_cooldown_ms.store(v);
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
    printf("[RTSP] 预绑定 DMA: %s (RTSP_PREBIND_DMA=%s)\n",
           rtsp_prebind_dma ? "开启" : "关闭",
           rtsp_prebind_dma ? "1" : "0");
    printf("[Tracker] 默认算法: %s | ReID: %s\n",
           rknn_lite::get_tracker_backend_name().c_str(),
           rknn_lite::resolve_tracker_reid_model().empty() ? "(none)" : rknn_lite::resolve_tracker_reid_model().c_str());
    printf("[AlertReport] 目标: http://%s:%d%s | cam0->%d cam1->%d | cooldown=%dms\n",
           g_report_server_host.c_str(), g_report_server_port, g_report_server_path.c_str(),
           g_report_camera_id_cam0, g_report_camera_id_cam1, g_box_count_alert_cooldown_ms.load());
    printf("[ForbiddenArea] 拉取: http://%s:%d%s | interval=%dms timeout=%dms | intrusion_cooldown=%dms\n",
           g_report_server_host.c_str(), g_report_server_port, g_forbidden_area_path.c_str(),
           g_forbidden_area_fetch_interval_ms.load(),
           g_forbidden_area_fetch_timeout_ms.load(),
           g_intrusion_alert_cooldown_ms.load());
    
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
#ifdef USE_RTSP_MPP
    std::vector<SlotBgrDmabuf> slot_output_buffers(total_slots);
#endif
    
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
    // 启动 CPU/NPU 使用率监控（每 10s 打印均值）
    std::thread usage_thread(usage_monitor_loop);

    refresh_forbidden_area_cache_if_needed(0);
    refresh_forbidden_area_cache_if_needed(1);
    
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
                        update_latest_frame_global(raw0, 0);
                        update_latest_frame_for_cam(0, raw0, 0);
                        {
                            std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                            if (g_rtsp_sender0 && g_rtsp_sender0->inited()) {
                                draw_rtsp_fps_overlay(raw0, g_rtsp_cam0_fps.load());
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
                        update_latest_frame_global(raw1, SLOTS_PER_CAM);
                        update_latest_frame_for_cam(1, raw1, SLOTS_PER_CAM);
                        {
                            std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                            if (g_rtsp_sender1 && g_rtsp_sender1->inited()) {
                                draw_rtsp_fps_overlay(raw1, g_rtsp_cam1_fps.load());
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
                    g_rtsp_cam0_fps.store((int)(push0 / elapsed));
                    g_rtsp_cam1_fps.store((int)(push1 / elapsed));
                    g_rtsp_video_fps.store(0);
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
            g_rtsp_cam0_fps.store(0);
            g_rtsp_cam1_fps.store(0);
            g_rtsp_video_fps.store(0);
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
                std::vector<DetectionResultItem> frame_detections;
                {
                    std::lock_guard<std::mutex> lock(g_model_mutex);
                    if (i < (int)rkpool.size() && rkpool[i]) {
                        worker_frame = rkpool[i]->ori_img;  // shallow copy: avoid clone in hot path
                        if (g_inference_enabled.load()) {
                            frame_detections = rkpool[i]->get_last_detections();
                            if (frame_detections.empty()) {
                                auto tracks = rkpool[i]->get_last_tracks();
                                if (!tracks.empty()) {
                                    frame_detections = build_detections_from_tracks(tracks);
                                }
                            }
                        }
                    }
                }
                if (worker_frame.empty()) {
                    slot_states[i].ready.store(false);
                    continue;
                }

                draw_forbidden_area_overlay_if_available(cam, worker_frame);
                draw_intrusion_boxes_overlay_if_needed(cam, worker_frame, frame_detections);
                
                // 更新最新帧（供 HTTP API 获取）
                // 视频模式只用 slot 0，摄像头模式用各自摄像头
                bool should_update_frame = false;
                if (g_video_mode.load()) {
                    should_update_frame = (i == 0);  // 视频模式只用 slot 0
                } else {
                    should_update_frame = true;  // 摄像头模式更新各自
                }
                
                if (should_update_frame) {
                    update_latest_frame_global(worker_frame, i);
                    if (!g_video_mode.load() && cam >= 0 && cam <= 1) {
                        update_latest_frame_for_cam(cam, worker_frame, i);
                    }
                }
                
                // RTSP 推流（绘制跟踪框）
#ifdef USE_RTSP_MPP
                if (g_rtsp_streaming.load()) {
                    cv::Mat frame_for_rtsp = worker_frame;  // shallow copy: avoid clone in hot path
                    int rtsp_fps_text = g_video_mode.load()
                        ? g_rtsp_video_fps.load()
                        : ((cam == 0) ? g_rtsp_cam0_fps.load() : g_rtsp_cam1_fps.load());
                    draw_rtsp_fps_overlay(frame_for_rtsp, rtsp_fps_text);

                    bool use_tracker = g_tracker_enabled.load();
                    if (use_tracker) {
                        // 注意：interf_detect_only() 已在 ori_img 上绘制跟踪框
                        // 这里若再次绘制会导致 RTSP 上出现“一人双框”
                        static int rtsp_skip_redraw_log_counter = 0;
                        if ((rtsp_skip_redraw_log_counter++ % 1000) == 0) {
                            printf("[RTSP] 跳过二次绘制（避免双框）\n");
                        }
                    }

                    {
                        std::lock_guard<std::mutex> rtsp_lock(g_rtsp_mutex);
                        if (g_video_mode.load() && g_rtsp_sender_video) {
                            bool pushed_ok = false;
                            if (i >= 0 && i < (int)slot_output_buffers.size()) {
                                SlotBgrDmabuf& out = slot_output_buffers[i];
                                if (out.fd >= 0 &&
                                    out.width == frame_for_rtsp.cols &&
                                    out.height == frame_for_rtsp.rows &&
                                    out.mat.data == frame_for_rtsp.data) {
                                    pushed_ok = g_rtsp_sender_video->push_bgr_dmabuf(
                                        out.fd, out.width, out.height, out.wstride, out.hstride);
                                }
                            }
                            if (!pushed_ok) {
                                pushed_ok = g_rtsp_sender_video->push(frame_for_rtsp);
                            }
                            if (pushed_ok) {
                                push0++;
                            } else {
                                printf("[RTSP] 视频推流写包失败，已自动停止，请重新调用 /api/rtsp/video/start\n");
                                g_rtsp_streaming = false;
                            }
                        } else if (!g_video_mode.load()) {
                            if (cam == 0 && g_rtsp_sender0) {
                                bool pushed_ok = false;
                                if (i >= 0 && i < (int)slot_output_buffers.size()) {
                                    SlotBgrDmabuf& out = slot_output_buffers[i];
                                    if (out.fd >= 0 &&
                                        out.width == frame_for_rtsp.cols &&
                                        out.height == frame_for_rtsp.rows &&
                                        out.mat.data == frame_for_rtsp.data) {
                                        pushed_ok = g_rtsp_sender0->push_bgr_dmabuf(
                                            out.fd, out.width, out.height, out.wstride, out.hstride);
                                    }
                                }
                                if (!pushed_ok) {
                                    pushed_ok = g_rtsp_sender0->push(frame_for_rtsp);
                                }
                                if (pushed_ok) {
                                    push0++;
                                } else {
                                    printf("[RTSP] Cam0 推流写包失败，已自动停止，请重新调用 /api/rtsp/start\n");
                                    g_rtsp_streaming = false;
                                }
                            }
                            if (cam == 1 && g_rtsp_sender1) {
                                bool pushed_ok = false;
                                if (i >= 0 && i < (int)slot_output_buffers.size()) {
                                    SlotBgrDmabuf& out = slot_output_buffers[i];
                                    if (out.fd >= 0 &&
                                        out.width == frame_for_rtsp.cols &&
                                        out.height == frame_for_rtsp.rows &&
                                        out.mat.data == frame_for_rtsp.data) {
                                        pushed_ok = g_rtsp_sender1->push_bgr_dmabuf(
                                            out.fd, out.width, out.height, out.wstride, out.hstride);
                                    }
                                }
                                if (!pushed_ok) {
                                    pushed_ok = g_rtsp_sender1->push(frame_for_rtsp);
                                }
                                if (pushed_ok) {
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
#ifdef USE_RTSP_MPP
            SlotBgrDmabuf* out_slot = nullptr;
            if (rtsp_prebind_dma && g_rtsp_streaming.load() && i >= 0 && i < (int)slot_output_buffers.size()) {
                out_slot = &slot_output_buffers[i];
            }
#endif

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
#ifdef USE_RTSP_MPP
                    if (out_slot) {
                        int vw = g_video_width.load();
                        int vh = g_video_height.load();
                        if (vw > 0 && vh > 0 && ensure_slot_bgr_dmabuf(out_slot, vw, vh)) {
                            frame = out_slot->mat;
                        }
                    }
#endif
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
#ifdef USE_RTSP_MPP
                if (out_slot) {
                    int cw = 0;
                    int ch = 0;
                    if (active_cap && active_cap->fd >= 0 && active_cap->streaming) {
                        cw = active_cap->width;
                        ch = active_cap->height;
                    } else if (active_cv_cap && active_cv_cap->isOpened()) {
                        cw = (int)active_cv_cap->get(cv::CAP_PROP_FRAME_WIDTH);
                        ch = (int)active_cv_cap->get(cv::CAP_PROP_FRAME_HEIGHT);
                    }
                    if (cw > 0 && ch > 0 && ensure_slot_bgr_dmabuf(out_slot, cw, ch)) {
                        frame = out_slot->mat;
                    }
                }
#endif
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
#ifdef USE_RTSP_MPP
            bool bound_dma_output = false;
            if (g_rtsp_streaming.load() && i >= 0 && i < (int)slot_output_buffers.size()) {
                SlotBgrDmabuf& out = slot_output_buffers[i];
                if (rtsp_prebind_dma) {
                    bool same_dma_mat = (out.fd >= 0 &&
                                         out.width == frame.cols &&
                                         out.height == frame.rows &&
                                         out.mat.data == frame.data);
                    if (same_dma_mat) {
                        worker->ori_img = frame;
                        bound_dma_output = true;
                    } else if (ensure_slot_bgr_dmabuf(&out, frame.cols, frame.rows)) {
                        if (out.mat.data != frame.data) {
                            frame.copyTo(out.mat);
                        }
                        worker->ori_img = out.mat;  // inference draw directly on DMA-mapped memory
                        bound_dma_output = true;
                    }
                } else {
                    if (ensure_slot_bgr_dmabuf(&out, frame.cols, frame.rows)) {
                        frame.copyTo(out.mat);
                        worker->ori_img = out.mat;  // inference draw directly on DMA-mapped memory
                        bound_dma_output = true;
                    }
                }
            }
            if (!bound_dma_output)
#endif
            {
                frame.copyTo(worker->ori_img);
            }
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

                    if (do_inference) {
                        // 更新对应摄像头的检测数量
                        int detection_count = worker->get_last_detection_count();
                        if (cam_id == 0) {
                            g_cam0_detection_count = detection_count;
                        } else {
                            g_cam1_detection_count = detection_count;
                        }

                        // 框数量超过阈值时主动上报（带当前帧截图）
                        maybe_trigger_detection_count_alert(cam_id, detection_count, worker->ori_img);

                        auto tracks = worker->get_last_tracks();
                        auto detections = worker->get_last_detections();
                        if (detections.empty() && !tracks.empty()) {
                            detections = build_detections_from_tracks(tracks);
                        }
                        maybe_trigger_forbidden_area_intrusion_alert(cam_id, detections, worker->ori_img);

                        // 如果启用了跟踪，保存跟踪结果
                        if (use_tracker) {
                            std::lock_guard<std::mutex> lock(g_tracker_mutex);
                            if (i < (int)g_tracker_results.size()) {
                                g_tracker_results[i] = tracks;
                            }
                        }
                    } else {
                        if (cam_id == 0) g_cam0_detection_count = 0;
                        else g_cam1_detection_count = 0;
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
            if (g_video_mode.load()) {
                g_rtsp_video_fps.store((int)(push0 / elapsed));
                g_rtsp_cam0_fps.store(0);
                g_rtsp_cam1_fps.store(0);
            } else {
                g_rtsp_cam0_fps.store((int)(push0 / elapsed));
                g_rtsp_cam1_fps.store((int)(push1 / elapsed));
                g_rtsp_video_fps.store(0);
            }
            
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
    if (usage_thread.joinable()) {
        usage_thread.join();
    }

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
    for (auto& out : slot_output_buffers) {
        release_slot_bgr_dmabuf(&out);
    }
#endif

    // 清理视频资源
    stop_video_reader();

    printf("[Main] 服务已退出\n");
    return 0;
}
