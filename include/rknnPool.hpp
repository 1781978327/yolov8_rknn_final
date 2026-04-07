#ifndef _RKNNPOOL_HPP
#define _RKNNPOOL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <cctype>
#include <unistd.h>
#include <libdrm/drm_fourcc.h>
#include <rockchip/mpp_buffer.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "rknn_api.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "deepsort.h"
#include "track.h"
#include "BYTETracker.h"

// 动态标签名称，从txt文件加载
static std::vector<std::string> coco_labels;
static std::string g_label_file_override;
static std::mutex g_label_file_mutex;
static std::string g_label_cache_key;

enum class TrackerBackend {
    ByteTrack,
    DeepSort
};

struct TrackerResultItem {
    int track_id = -1;
    int label = -1;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    bool active = false;
    std::vector<std::pair<float, float>> trajectory;
};

static TrackerBackend g_tracker_backend_override = TrackerBackend::ByteTrack;
static std::string g_tracker_reid_model_override;
static int g_deepsort_skip_frames = 0;
static std::mutex g_tracker_config_mutex;
static DeepSort* g_shared_deepsort_trackers[3] = {nullptr, nullptr, nullptr};
static std::mutex g_shared_deepsort_init_mutex;
static std::mutex g_shared_deepsort_runtime_mutex[3];
static std::string g_shared_deepsort_model_path;
static std::atomic<long long> g_shared_deepsort_frame_counter[3];
static std::vector<DetectBox> g_shared_deepsort_last_detections[3];
static BYTETracker* g_shared_bytetrack_trackers[3] = {nullptr, nullptr, nullptr};
static std::mutex g_shared_bytetrack_init_mutex;
static std::mutex g_shared_bytetrack_runtime_mutex[3];
static std::atomic<long long> g_shared_bytetrack_frame_counter[3];

static bool is_readable_file(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

static std::string normalize_tracker_backend_name(const std::string& backend) {
    std::string normalized = backend;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return (char)std::tolower(ch); });
    return normalized;
}

static std::string tracker_backend_to_string(TrackerBackend backend) {
    return backend == TrackerBackend::DeepSort ? "deepsort" : "bytetrack";
}

static bool parse_tracker_backend_name(const std::string& backend_name, TrackerBackend* backend_out) {
    std::string normalized = normalize_tracker_backend_name(backend_name);
    if (normalized.empty() || normalized == "bytetrack" || normalized == "byte" || normalized == "bt") {
        if (backend_out) *backend_out = TrackerBackend::ByteTrack;
        return true;
    }
    if (normalized == "deepsort" || normalized == "deep") {
        if (backend_out) *backend_out = TrackerBackend::DeepSort;
        return true;
    }
    return false;
}

static std::string resolve_reid_model_path_locked() {
    if (is_readable_file(g_tracker_reid_model_override)) {
        return g_tracker_reid_model_override;
    }

    const char* candidates[] = {
        "/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn",
        "../model/RK3588/osnet_x0_25_market.rknn",
        "./model/RK3588/osnet_x0_25_market.rknn"
    };
    for (const char* candidate : candidates) {
        if (is_readable_file(candidate)) {
            return candidate;
        }
    }
    return "";
}

static cv::Scalar tracker_color_from_id(int track_id) {
    unsigned int seed = (unsigned int)(track_id < 0 ? -track_id : track_id);
    int b = 80 + (seed * 37) % 176;
    int g = 80 + (seed * 57) % 176;
    int r = 80 + (seed * 97) % 176;
    return cv::Scalar(b, g, r);
}

static int clamp_coord_int(float value, int max_value) {
    int ivalue = (int)value;
    if (ivalue < 0) return 0;
    if (ivalue > max_value) return max_value;
    return ivalue;
}

static int sanitize_deepsort_skip_frames(int skip_frames) {
    if (skip_frames < 0) return 0;
    if (skip_frames > 10) return 10;
    return skip_frames;
}

// FPS 计算变量
static int fps_frame_count = 0;
static double fps_last_time = 0.0;
static double current_fps = 0.0;
static std::atomic<int> g_preprocess_rgb_iomem_log_counter(0);
static std::atomic<int> g_preprocess_video_iomem_log_counter(0);
static std::atomic<int> g_preprocess_legacy_log_counter(0);
static std::atomic<int> g_preprocess_video_iomem_fail_counter(0);
static std::atomic<int> g_preprocess_video_iomem_fallback_counter(0);
static std::atomic<bool> g_preprocess_video_iomem_disabled(false);

static bool load_labels_from_txt(const std::string& label_path) {
    std::ifstream file(label_path);
    if (!file.is_open()) {
        return false;
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            labels.push_back(line.substr(start, end - start + 1));
        }
    }
    file.close();

    if (labels.empty()) {
        return false;
    }

    coco_labels = labels;
    OBJ_CLASS_NUM = (int)coco_labels.size();
    printf("加载标签文件: %s (%d 类)\n", label_path.c_str(), OBJ_CLASS_NUM);
    return true;
}

// 加载标签文件：自动检测模型对应的标签文件
// yolov8n/yolov8s/yolov8m → 使用 coco_80_labels_list.txt
// 行人摔倒.rknn → 使用同目录下的 Untitled 文件或默认4类标签
static void load_labels(const char* model_path) {
    std::string model_name = model_path ? model_path : "";
    size_t pos = model_name.find_last_of("/\\");
    if (pos != std::string::npos) {
        model_name = model_name.substr(pos + 1);
    }

    std::string override_path;
    {
        std::lock_guard<std::mutex> lock(g_label_file_mutex);
        override_path = g_label_file_override;
    }
    const std::string cache_key = model_name + "|" + override_path;
    if (cache_key == g_label_cache_key && !coco_labels.empty()) {
        OBJ_CLASS_NUM = (int)coco_labels.size();
        return;
    }

    if (!override_path.empty()) {
        if (load_labels_from_txt(override_path)) {
            g_label_cache_key = cache_key;
            return;
        }
        printf("警告: 指定标签文件不可用，回退默认标签: %s\n", override_path.c_str());
    }

    // 默认优先使用 coco 标签 txt
    if (load_labels_from_txt("/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/coco_80_labels_list.txt") ||
        load_labels_from_txt("../model/coco_80_labels_list.txt") ||
        load_labels_from_txt("./model/coco_80_labels_list.txt")) {
        g_label_cache_key = cache_key;
        return;
    }

    if (model_name.find("行人摔倒") != std::string::npos ||
        model_name.find("摔倒") != std::string::npos) {
        // 行人摔倒模型使用4类标签
        coco_labels = {"person", "fall", "fight", "armed"};
    } else {
        // 默认80类coco标签（硬编码兜底）
        coco_labels = {
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };
    }
    
    OBJ_CLASS_NUM = (int)coco_labels.size();
    printf("加载标签文件: %d 个类别\n", OBJ_CLASS_NUM);
    g_label_cache_key = cache_key;
}

class rknn_lite {
private:
    struct VideoDmabufFrame {
        bool valid = false;
        int fd = -1;
        int size = 0;
        int width = 0;
        int height = 0;
        int wstride = 0;
        int hstride = 0;
        uint32_t drm_format = 0;
    };

    struct VideoRgbStageBuffer {
        MppBuffer buffer = nullptr;
        int fd = -1;
        void* ptr = nullptr;
        int size = 0;
        int width = 0;
        int height = 0;
        int wstride = 0;
        int hstride = 0;
    };

    rknn_app_context_t app_ctx;  // RKNN应用上下文
    int ret;  // 函数返回值
    BYTETracker* bytetrack_;  // BYTETracker 跟踪器
    TrackerBackend tracker_backend_;  // 当前跟踪算法
    std::string reid_model_path_;  // DeepSORT ReID 模型
    long long frame_count_;  // 帧计数器
    bool enable_tracker_;  // 是否启用跟踪
    bool use_tracker_this_frame_;  // 本帧是否使用跟踪器
    int tracker_stream_id_;  // 跟踪流ID: 0=cam0,1=cam1,2=video
    bool last_tracker_feature_match_;  // 当前帧是否执行了特征匹配
    rknn_tensor_mem* input_mem_;  // RKNN input io_mem
    bool input_mem_ready_;  // 是否已启用 io_mem 输入
    VideoDmabufFrame video_dmabuf_frame_;  // 视频模式 DRM PRIME 帧
    VideoRgbStageBuffer video_rgb_stage_buffer_;  // 视频模式 fd-backed RGA 中转 RGB 帧
    cv::Mat video_nv12_frame_;  // 视频模式 packed NV12 帧
    bool video_nv12_valid_;  // 是否存在 packed NV12 帧
    int video_nv12_width_;  // packed NV12 逻辑宽度
    int video_nv12_height_;  // packed NV12 逻辑高度
    int video_nv12_wstride_;  // packed NV12 stride
    int video_nv12_hstride_;  // packed NV12 height stride
    static std::atomic<float> detection_threshold;  // 置信度阈值

public:
    cv::Mat ori_img;  // 原始图像
    static std::atomic<int> last_detection_count;  // 最后检测到的目标数量

    rknn_lite(char* model_path, int core_id);
    ~rknn_lite();
    int interf();
    int interf_detect_only();  // 仅推理，不绘制
    void set_use_tracker(bool use);  // 设置是否使用跟踪器
    void set_video_dmabuf_frame(int fd, int size, int width, int height, int wstride, int hstride, uint32_t drm_format);
    void set_video_nv12_frame(const cv::Mat& nv12_frame, int width, int height, int wstride, int hstride);
    void clear_video_dmabuf_frame();
    bool is_tracker_enabled() const { return enable_tracker_; }
    std::vector<TrackerResultItem> get_last_tracks() const { return last_tracks_; }
    static void set_detection_threshold(float threshold);  // 设置置信度阈值
    static float get_detection_threshold();  // 获取置信度阈值
    static void set_label_file_override(const std::string& label_file);  // 设置标签txt覆盖路径，空字符串=默认
    static std::string get_label_file_override();  // 获取标签txt覆盖路径
    static bool set_tracker_backend(const std::string& backend_name);  // 设置跟踪算法
    static std::string get_tracker_backend_name();  // 获取当前跟踪算法名称
    static void set_tracker_reid_model_override(const std::string& model_file);  // 设置 DeepSORT ReID 模型
    static std::string get_tracker_reid_model_override();  // 获取 DeepSORT ReID 模型
    static std::string resolve_tracker_reid_model();  // 解析有效的 DeepSORT ReID 模型路径
    static void set_deepsort_skip_frames(int skip_frames);  // 设置 DeepSORT 跳帧数
    static int get_deepsort_skip_frames();  // 获取 DeepSORT 跳帧数
    static void reset_shared_deepsort_trackers();  // 重置共享 DeepSORT 状态
    static void reset_shared_bytetrack_trackers();  // 重置共享 ByteTrack 状态
    void set_tracker_stream_id(int stream_id);  // 设置当前worker所属的跟踪流

    int RGA_bgr_to_rgb(const cv::Mat& bgr_image, cv::Mat &rgb_image);
    int RGA_resize(const cv::Mat& src, cv::Mat& dst, int dst_width, int dst_height);

private:
    bool upload_rgb_to_input_mem(const cv::Mat& rgb_image);
    bool upload_rgb_fd_to_input_mem(int fd, int width, int height, int wstride, int hstride);
    void release_video_rgb_stage_buffer();
    bool ensure_video_rgb_stage_buffer(int width, int height);
    bool preprocess_video_dmabuf_via_rgb_stage();
    bool preprocess_video_dmabuf_to_input_mem();
    bool preprocess_video_nv12_to_input_mem();
    int draw_tracker_results(cv::Mat& img, const std::vector<TrackerResultItem>& tracks);
    int draw_unmatched_detections(cv::Mat& img, const object_detect_result_list& od_results,
                                  const std::vector<TrackerResultItem>& tracks);
    std::vector<TrackerResultItem> update_bytetrack(const object_detect_result_list& od_results);
    std::vector<TrackerResultItem> update_deepsort(cv::Mat& img, const object_detect_result_list& od_results);
    BYTETracker* ensure_shared_bytetrack_tracker();
    DeepSort* ensure_shared_deepsort_tracker();

private:
    std::vector<TrackerResultItem> last_tracks_;  // 最后一帧的跟踪结果
};

// 静态成员初始化
std::atomic<float> rknn_lite::detection_threshold(0.5f);
std::atomic<int> rknn_lite::last_detection_count(0);

// 构造函数：初始化RKNN模型
rknn_lite::rknn_lite(char* model_path, int core_id) {
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));
    bytetrack_ = nullptr;
    frame_count_ = 0;
    use_tracker_this_frame_ = false;
    tracker_stream_id_ = 0;
    last_tracker_feature_match_ = true;
    input_mem_ = nullptr;
    input_mem_ready_ = false;
    video_rgb_stage_buffer_ = VideoRgbStageBuffer{};
    video_nv12_valid_ = false;
    video_nv12_width_ = 0;
    video_nv12_height_ = 0;
    video_nv12_wstride_ = 0;
    video_nv12_hstride_ = 0;

    // 总是启用跟踪器（运行时可通过 API 控制开关）
    enable_tracker_ = true;

    {
        std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
        tracker_backend_ = g_tracker_backend_override;
        reid_model_path_ = resolve_reid_model_path_locked();
    }

    // 加载模型文件
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_path, model_data_size);
    
    // 加载标签文件（支持运行时覆盖）
    load_labels(model_path);
    
    // 初始化RKNN上下文
    ret = rknn_init(&app_ctx.rknn_ctx, model_data, model_data_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("rknn_init 错误 ret=%d\n", ret);
        exit(-1);
    }

    // 设置NPU核心掩码
    rknn_core_mask core_mask;
    switch(core_id % 3) {
        case 0: core_mask = RKNN_NPU_CORE_0; break;
        case 1: core_mask = RKNN_NPU_CORE_1; break;
        default: core_mask = RKNN_NPU_CORE_2;
    }

    if (enable_tracker_) {
        if (tracker_backend_ == TrackerBackend::DeepSort) {
            if (!reid_model_path_.empty() && is_readable_file(reid_model_path_)) {
                printf("DeepSORT 跟踪已启用，共享 ReID 模型: %s\n", reid_model_path_.c_str());
            } else {
                printf("警告: DeepSORT 所需 ReID 模型不可用，已回退为 BYTETracker\n");
                tracker_backend_ = TrackerBackend::ByteTrack;
            }
        }

        if (tracker_backend_ == TrackerBackend::ByteTrack) {
            const char* trail_sec_env = getenv("TRACK_TRAIL_SECONDS");
            int trail_seconds = 3;
            if (trail_sec_env) {
                trail_seconds = atoi(trail_sec_env);
                if (trail_seconds <= 0) trail_seconds = 1;
            }
            int trail_frames = trail_seconds * 25;
            STrack::set_max_trail_length(trail_frames);

            printf("BYTETracker 跟踪已启用，轨迹保留 %d 秒（约 %d 帧）\n", trail_seconds, trail_frames);
        }
    }

    ret = rknn_set_core_mask(app_ctx.rknn_ctx, core_mask);
    if (ret < 0) {
        printf("rknn_set_core_mask 错误 ret=%d\n", ret);
        exit(-1);
    }

    // 获取模型输入输出信息
    ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx.io_num, sizeof(app_ctx.io_num));
    if (ret < 0) {
        printf("rknn_query io_num 错误 ret=%d\n", ret);
        exit(-1);
    }

    // 获取输入属性
    app_ctx.input_attrs = new rknn_tensor_attr[app_ctx.io_num.n_input];
    memset(app_ctx.input_attrs, 0, sizeof(rknn_tensor_attr) * app_ctx.io_num.n_input);
    for (uint32_t i = 0; i < app_ctx.io_num.n_input; i++) {
        app_ctx.input_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx.input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query input_attrs 错误 ret=%d\n", ret);
            exit(-1);
        }
    }

    // 获取输出属性
    app_ctx.output_attrs = new rknn_tensor_attr[app_ctx.io_num.n_output];
    memset(app_ctx.output_attrs, 0, sizeof(rknn_tensor_attr) * app_ctx.io_num.n_output);
    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        app_ctx.output_attrs[i].index = i;
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx.output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query output_attrs 错误 ret=%d\n", ret);
            exit(-1);
        }
    }

    // 检查量化类型
    if (app_ctx.output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && 
        app_ctx.output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx.is_quant = true;
    } else {
        app_ctx.is_quant = false;
    }

    // 设置模型维度
    if (app_ctx.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_height = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[3];
    } else {
        app_ctx.model_height = app_ctx.input_attrs[0].dims[1];
        app_ctx.model_width = app_ctx.input_attrs[0].dims[2];
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[3];
    }

    app_ctx.input_attrs[0].type = RKNN_TENSOR_UINT8;
    app_ctx.input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    app_ctx.input_attrs[0].pass_through = 0;
    if (app_ctx.input_attrs[0].w_stride == 0) app_ctx.input_attrs[0].w_stride = app_ctx.model_width;
    if (app_ctx.input_attrs[0].h_stride == 0) app_ctx.input_attrs[0].h_stride = app_ctx.model_height;

    input_mem_ = rknn_create_mem(app_ctx.rknn_ctx, app_ctx.input_attrs[0].size_with_stride);
    if (input_mem_) {
        int mem_ret = rknn_set_io_mem(app_ctx.rknn_ctx, input_mem_, &app_ctx.input_attrs[0]);
        if (mem_ret == RKNN_SUCC) {
            input_mem_ready_ = true;
        } else {
            rknn_destroy_mem(app_ctx.rknn_ctx, input_mem_);
            input_mem_ = nullptr;
            input_mem_ready_ = false;
            printf("警告: rknn_set_io_mem input 失败，回退 rknn_inputs_set ret=%d\n", mem_ret);
        }
    } else {
        printf("警告: rknn_create_mem input 失败，回退 rknn_inputs_set\n");
    }
}

// 析构函数：释放资源
rknn_lite::~rknn_lite() {
    release_video_rgb_stage_buffer();
    if (input_mem_ != nullptr && app_ctx.rknn_ctx != 0) {
        rknn_destroy_mem(app_ctx.rknn_ctx, input_mem_);
        input_mem_ = nullptr;
    }
    if (app_ctx.rknn_ctx != 0) {
        rknn_destroy(app_ctx.rknn_ctx);
        app_ctx.rknn_ctx = 0;
    }
    if (app_ctx.input_attrs != nullptr) {
        delete[] app_ctx.input_attrs;
        app_ctx.input_attrs = nullptr;
    }
    if (app_ctx.output_attrs != nullptr) {
        delete[] app_ctx.output_attrs;
        app_ctx.output_attrs = nullptr;
    }
    clear_video_dmabuf_frame();
    if (bytetrack_ != nullptr) {
        delete bytetrack_;
        bytetrack_ = nullptr;
    }
}

// BGR转RGB函数（使用RGA加速）
int rknn_lite::RGA_bgr_to_rgb(const cv::Mat& bgr_image, cv::Mat &rgb_image) {
    // 创建输出图像
    rgb_image.create(bgr_image.size(), bgr_image.type());
    
    rga_buffer_t src_img, dst_img;
    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    // 设置输入输出参数
    int src_width = bgr_image.cols;
    int src_height = bgr_image.rows;
    int dst_width = rgb_image.cols;
    int dst_height = rgb_image.rows;

    // 设置图像格式
    int src_format = RK_FORMAT_BGR_888;
    int dst_format = RK_FORMAT_RGB_888;

    // 包装图像数据到RGA缓冲区
    src_img = wrapbuffer_virtualaddr((void *)bgr_image.data, src_width, src_height, src_format);
    dst_img = wrapbuffer_virtualaddr((void *)rgb_image.data, dst_width, dst_height, dst_format);

    // 执行颜色空间转换
    IM_STATUS status = imcvtcolor(src_img, dst_img, src_format, dst_format);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA BGR转RGB错误: %s\n", imStrError(status));
        return -1;
    }
    
    return 0;
}

// 图像缩放函数（使用RGA加速）
int rknn_lite::RGA_resize(const cv::Mat& src, cv::Mat& dst, int dst_width, int dst_height) {
    // 创建目标图像
    dst.create(dst_height, dst_width, src.type());
    
    rga_buffer_t src_img, dst_img;
    im_rect src_rect, dst_rect;
    
    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    // 设置图像格式
    int format = RK_FORMAT_RGB_888;  

    // 包装图像数据
    src_img = wrapbuffer_virtualaddr((void*)src.data, src.cols, src.rows, format);
    dst_img = wrapbuffer_virtualaddr((void*)dst.data, dst.cols, dst.rows, format);

    // 执行缩放操作
    IM_STATUS status = imresize(src_img, dst_img);
    if (status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "RGA缩放错误: %s\n", imStrError(status));
        return -1;
    }
    
    return 0;
}

void rknn_lite::set_video_dmabuf_frame(int fd, int size, int width, int height, int wstride, int hstride, uint32_t drm_format) {
    clear_video_dmabuf_frame();
    if (fd < 0 || size <= 0 || width <= 0 || height <= 0) {
        return;
    }
    video_dmabuf_frame_.valid = true;
    video_dmabuf_frame_.fd = fd;
    video_dmabuf_frame_.size = size;
    video_dmabuf_frame_.width = width;
    video_dmabuf_frame_.height = height;
    video_dmabuf_frame_.wstride = wstride;
    video_dmabuf_frame_.hstride = hstride;
    video_dmabuf_frame_.drm_format = drm_format;
}

void rknn_lite::set_video_nv12_frame(const cv::Mat& nv12_frame, int width, int height, int wstride, int hstride) {
    if (nv12_frame.empty() || width <= 0 || height <= 0) {
        return;
    }
    video_nv12_frame_ = nv12_frame.clone();
    video_nv12_valid_ = !video_nv12_frame_.empty();
    video_nv12_width_ = width;
    video_nv12_height_ = height;
    video_nv12_wstride_ = wstride > 0 ? wstride : width;
    video_nv12_hstride_ = hstride > 0 ? hstride : height;
}

void rknn_lite::clear_video_dmabuf_frame() {
    if (video_dmabuf_frame_.fd >= 0) {
        close(video_dmabuf_frame_.fd);
    }
    video_dmabuf_frame_ = VideoDmabufFrame{};
    video_nv12_frame_.release();
    video_nv12_valid_ = false;
    video_nv12_width_ = 0;
    video_nv12_height_ = 0;
    video_nv12_wstride_ = 0;
    video_nv12_hstride_ = 0;
}

bool rknn_lite::upload_rgb_to_input_mem(const cv::Mat& rgb_image) {
    if (!input_mem_ready_ || !input_mem_ || rgb_image.empty()) {
        return false;
    }

    const int dst_w = app_ctx.model_width;
    const int dst_h = app_ctx.model_height;
    const int dst_w_stride = app_ctx.input_attrs[0].w_stride > 0 ? (int)app_ctx.input_attrs[0].w_stride : dst_w;
    const int dst_h_stride = app_ctx.input_attrs[0].h_stride > 0 ? (int)app_ctx.input_attrs[0].h_stride : dst_h;

    rga_buffer_t src = wrapbuffer_virtualaddr((void*)rgb_image.data, rgb_image.cols, rgb_image.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst;
    if (input_mem_->fd > 0) {
        dst = wrapbuffer_fd(input_mem_->fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        dst = wrapbuffer_virtualaddr(input_mem_->virt_addr, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    }

    IM_STATUS status = IM_STATUS_SUCCESS;
    if (rgb_image.cols != dst_w || rgb_image.rows != dst_h) {
        status = imresize(src, dst);
    } else {
        status = improcess(src, dst,
                           wrapbuffer_virtualaddr(nullptr, 0, 0, RK_FORMAT_RGBA_8888),
                           im_rect{0, 0, rgb_image.cols, rgb_image.rows},
                           im_rect{0, 0, dst_w, dst_h},
                           im_rect{0, 0, 0, 0}, IM_SYNC);
    }
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        return false;
    }

    int log_idx = g_preprocess_rgb_iomem_log_counter.fetch_add(1) + 1;
    if ((log_idx % 120) == 0) {
        printf("[Preprocess] rgb_mat -> RGA -> input_mem (%dx%d -> %dx%d, stride=%dx%d)\n",
               rgb_image.cols, rgb_image.rows,
               dst_w, dst_h, dst_w_stride, dst_h_stride);
    }
    return true;
}

bool rknn_lite::upload_rgb_fd_to_input_mem(int fd, int width, int height, int wstride, int hstride) {
    if (!input_mem_ready_ || !input_mem_ || fd < 0 || width <= 0 || height <= 0) {
        return false;
    }

    const int dst_w = app_ctx.model_width;
    const int dst_h = app_ctx.model_height;
    const int dst_w_stride = app_ctx.input_attrs[0].w_stride > 0 ? (int)app_ctx.input_attrs[0].w_stride : dst_w;
    const int dst_h_stride = app_ctx.input_attrs[0].h_stride > 0 ? (int)app_ctx.input_attrs[0].h_stride : dst_h;

    rga_buffer_t src = wrapbuffer_fd(fd, width, height, RK_FORMAT_RGB_888,
                                     wstride > 0 ? wstride : width,
                                     hstride > 0 ? hstride : height);
    rga_buffer_t dst;
    if (input_mem_->fd > 0) {
        dst = wrapbuffer_fd(input_mem_->fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        dst = wrapbuffer_virtualaddr(input_mem_->virt_addr, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    }

    IM_STATUS status = IM_STATUS_SUCCESS;
    if (width != dst_w || height != dst_h) {
        status = imresize(src, dst);
    } else {
        status = improcess(src, dst,
                           wrapbuffer_virtualaddr(nullptr, 0, 0, RK_FORMAT_RGBA_8888),
                           im_rect{0, 0, width, height},
                           im_rect{0, 0, dst_w, dst_h},
                           im_rect{0, 0, 0, 0}, IM_SYNC);
    }
    return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR;
}

void rknn_lite::release_video_rgb_stage_buffer() {
    if (video_rgb_stage_buffer_.buffer) {
        mpp_buffer_put(video_rgb_stage_buffer_.buffer);
    }
    video_rgb_stage_buffer_ = VideoRgbStageBuffer{};
}

bool rknn_lite::ensure_video_rgb_stage_buffer(int width, int height) {
    if (width <= 0 || height <= 0) return false;

    if (video_rgb_stage_buffer_.buffer &&
        video_rgb_stage_buffer_.width == width &&
        video_rgb_stage_buffer_.height == height &&
        video_rgb_stage_buffer_.wstride == width &&
        video_rgb_stage_buffer_.hstride == height &&
        video_rgb_stage_buffer_.fd >= 0) {
        return true;
    }

    release_video_rgb_stage_buffer();

    const int stage_size = width * height * 3;
    MPP_RET ret = mpp_buffer_get(nullptr, &video_rgb_stage_buffer_.buffer, stage_size);
    if (ret != MPP_OK || !video_rgb_stage_buffer_.buffer) {
        release_video_rgb_stage_buffer();
        return false;
    }

    video_rgb_stage_buffer_.fd = mpp_buffer_get_fd(video_rgb_stage_buffer_.buffer);
    video_rgb_stage_buffer_.ptr = mpp_buffer_get_ptr(video_rgb_stage_buffer_.buffer);
    video_rgb_stage_buffer_.size = stage_size;
    video_rgb_stage_buffer_.width = width;
    video_rgb_stage_buffer_.height = height;
    video_rgb_stage_buffer_.wstride = width;
    video_rgb_stage_buffer_.hstride = height;

    if (video_rgb_stage_buffer_.fd < 0) {
        release_video_rgb_stage_buffer();
        return false;
    }

    return true;
}

bool rknn_lite::preprocess_video_dmabuf_via_rgb_stage() {
    if (!video_dmabuf_frame_.valid || video_dmabuf_frame_.fd < 0 ||
        video_dmabuf_frame_.width <= 0 || video_dmabuf_frame_.height <= 0 ||
        video_dmabuf_frame_.wstride <= 0 || video_dmabuf_frame_.hstride <= 0 ||
        video_dmabuf_frame_.drm_format != DRM_FORMAT_NV12) {
        return false;
    }

    if (!ensure_video_rgb_stage_buffer(video_dmabuf_frame_.width, video_dmabuf_frame_.height)) {
        return false;
    }

    rga_buffer_t src = wrapbuffer_fd(video_dmabuf_frame_.fd,
                                     video_dmabuf_frame_.width,
                                     video_dmabuf_frame_.height,
                                     RK_FORMAT_YCbCr_420_SP,
                                     video_dmabuf_frame_.wstride,
                                     video_dmabuf_frame_.hstride);
    rga_buffer_t stage = wrapbuffer_fd(video_rgb_stage_buffer_.fd,
                                       video_rgb_stage_buffer_.width,
                                       video_rgb_stage_buffer_.height,
                                       RK_FORMAT_RGB_888,
                                       video_rgb_stage_buffer_.wstride,
                                       video_rgb_stage_buffer_.hstride);

    IM_STATUS status = imcvtcolor(src, stage, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        return false;
    }

    int log_idx = g_preprocess_video_iomem_log_counter.fetch_add(1) + 1;
    if ((log_idx % 120) == 0) {
        printf("[Preprocess] video drm_fd -> RGA csc -> rgb_stage (%dx%d stride=%dx%d)\n",
               video_dmabuf_frame_.width, video_dmabuf_frame_.height,
               video_dmabuf_frame_.wstride, video_dmabuf_frame_.hstride);
    }

    return upload_rgb_fd_to_input_mem(video_rgb_stage_buffer_.fd,
                                      video_rgb_stage_buffer_.width,
                                      video_rgb_stage_buffer_.height,
                                      video_rgb_stage_buffer_.wstride,
                                      video_rgb_stage_buffer_.hstride);
}

bool rknn_lite::preprocess_video_dmabuf_to_input_mem() {
    if (g_preprocess_video_iomem_disabled.load()) {
        return preprocess_video_dmabuf_via_rgb_stage();
    }

    auto log_video_iomem_fail = [&](const char* reason, int extra = 0) {
        int fail_idx = g_preprocess_video_iomem_fail_counter.fetch_add(1) + 1;
        if (fail_idx <= 10 || (fail_idx % 120) == 0) {
            printf("[Preprocess] video drm_fd -> input_mem failed: %s (fd=%d size=%d fmt=0x%08x %dx%d stride=%dx%d extra=%d)\n",
                   reason,
                   video_dmabuf_frame_.fd,
                   video_dmabuf_frame_.size,
                   video_dmabuf_frame_.drm_format,
                   video_dmabuf_frame_.width,
                   video_dmabuf_frame_.height,
                   video_dmabuf_frame_.wstride,
                   video_dmabuf_frame_.hstride,
                   extra);
        }
    };
    auto disable_video_iomem = [&](const char* reason) {
        bool expected = false;
        if (g_preprocess_video_iomem_disabled.compare_exchange_strong(expected, true)) {
            printf("[Preprocess] disable direct video drm_fd -> input_mem after %s, fallback to two-stage RGA path\n",
                   reason);
        }
    };

    if (!input_mem_ready_ || !input_mem_ || !video_dmabuf_frame_.valid) {
        log_video_iomem_fail("invalid-input");
        return false;
    }
    if (video_dmabuf_frame_.drm_format != DRM_FORMAT_NV12) {
        log_video_iomem_fail("unsupported-drm-format");
        return false;
    }

    const int dst_w = app_ctx.model_width;
    const int dst_h = app_ctx.model_height;
    const int dst_w_stride = app_ctx.input_attrs[0].w_stride > 0 ? (int)app_ctx.input_attrs[0].w_stride : dst_w;
    const int dst_h_stride = app_ctx.input_attrs[0].h_stride > 0 ? (int)app_ctx.input_attrs[0].h_stride : dst_h;

    rga_buffer_t src = wrapbuffer_fd(video_dmabuf_frame_.fd,
                                     video_dmabuf_frame_.width,
                                     video_dmabuf_frame_.height,
                                     RK_FORMAT_YCbCr_420_SP,
                                     video_dmabuf_frame_.wstride,
                                     video_dmabuf_frame_.hstride);

    rga_buffer_t dst;
    if (input_mem_->fd > 0) {
        dst = wrapbuffer_fd(input_mem_->fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        dst = wrapbuffer_virtualaddr(input_mem_->virt_addr, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    }

    im_rect src_rect = {0, 0, video_dmabuf_frame_.width, video_dmabuf_frame_.height};
    im_rect dst_rect = {0, 0, dst_w, dst_h};
    int check = imcheck(src, dst, src_rect, dst_rect);
    if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR) {
        log_video_iomem_fail("imcheck", check);
        disable_video_iomem("imcheck");
        return preprocess_video_dmabuf_via_rgb_stage();
    }

    IM_STATUS status = improcess(src, dst,
                                 wrapbuffer_virtualaddr(nullptr, 0, 0, RK_FORMAT_RGBA_8888),
                                 im_rect{0, 0, video_dmabuf_frame_.width, video_dmabuf_frame_.height},
                                 im_rect{0, 0, dst_w, dst_h},
                                 im_rect{0, 0, 0, 0}, IM_SYNC);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        log_video_iomem_fail("improcess", status);
        disable_video_iomem("improcess");
        return preprocess_video_dmabuf_via_rgb_stage();
    }

    int log_idx = g_preprocess_video_iomem_log_counter.fetch_add(1) + 1;
    if ((log_idx % 120) == 0) {
        printf("[Preprocess] video drm_fd -> RGA -> input_mem (fd=%d, %dx%d, stride=%dx%d -> %dx%d)\n",
               video_dmabuf_frame_.fd,
               video_dmabuf_frame_.width, video_dmabuf_frame_.height,
               video_dmabuf_frame_.wstride, video_dmabuf_frame_.hstride,
               dst_w, dst_h);
    }
    return true;
}

bool rknn_lite::preprocess_video_nv12_to_input_mem() {
    if (!input_mem_ready_ || !input_mem_ || !video_nv12_valid_ || video_nv12_frame_.empty() ||
        video_nv12_width_ <= 0 || video_nv12_height_ <= 0) {
        return false;
    }

    const int dst_w = app_ctx.model_width;
    const int dst_h = app_ctx.model_height;
    const int dst_w_stride = app_ctx.input_attrs[0].w_stride > 0 ? (int)app_ctx.input_attrs[0].w_stride : dst_w;
    const int dst_h_stride = app_ctx.input_attrs[0].h_stride > 0 ? (int)app_ctx.input_attrs[0].h_stride : dst_h;

    rga_buffer_t src = wrapbuffer_virtualaddr(video_nv12_frame_.data, video_nv12_width_, video_nv12_height_, RK_FORMAT_YCbCr_420_SP);
    src.wstride = video_nv12_wstride_;
    src.hstride = video_nv12_hstride_;

    rga_buffer_t dst;
    if (input_mem_->fd > 0) {
        dst = wrapbuffer_fd(input_mem_->fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    } else {
        dst = wrapbuffer_virtualaddr(input_mem_->virt_addr, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_stride, dst_h_stride);
    }

    im_rect src_rect = {0, 0, video_nv12_wstride_, video_nv12_hstride_};
    im_rect dst_rect = {0, 0, dst_w, dst_h};
    int check = imcheck(src, dst, src_rect, dst_rect);
    if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR) {
        int fail_idx = g_preprocess_video_iomem_fail_counter.fetch_add(1) + 1;
        if (fail_idx <= 10 || (fail_idx % 120) == 0) {
            printf("[Preprocess] video nv12 -> input_mem failed: imcheck (%dx%d stride=%dx%d -> %dx%d extra=%d)\n",
                   video_nv12_width_, video_nv12_height_,
                   video_nv12_wstride_, video_nv12_hstride_, dst_w, dst_h, check);
        }
        return false;
    }

    IM_STATUS status = improcess(src, dst,
                                 wrapbuffer_virtualaddr(nullptr, 0, 0, RK_FORMAT_RGBA_8888),
                                 src_rect,
                                 dst_rect,
                                 im_rect{0, 0, 0, 0}, IM_SYNC);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        int fail_idx = g_preprocess_video_iomem_fail_counter.fetch_add(1) + 1;
        if (fail_idx <= 10 || (fail_idx % 120) == 0) {
            printf("[Preprocess] video nv12 -> input_mem failed: improcess status=%s\n", imStrError(status));
        }
        return false;
    }

    int log_idx = g_preprocess_video_iomem_log_counter.fetch_add(1) + 1;
    if ((log_idx % 120) == 0) {
        printf("[Preprocess] video nv12 -> RGA -> input_mem (%dx%d stride=%dx%d -> %dx%d)\n",
               video_nv12_width_, video_nv12_height_,
               video_nv12_wstride_, video_nv12_hstride_,
               dst_w, dst_h);
    }
    return true;
}

int rknn_lite::draw_tracker_results(cv::Mat& img, const std::vector<TrackerResultItem>& tracks) {
    char text[256];
    int drawn = 0;
    const int min_dim = std::min(img.cols, img.rows);
    const int box_thickness = std::max(2, min_dim / 360);
    const int trail_thickness = std::max(2, min_dim / 540);
    const double font_scale = std::max(0.7, min_dim / 900.0);
    const int font_thickness = std::max(2, min_dim / 700);
    for (const auto& track : tracks) {
        if (!track.active || track.track_id < 0) continue;

        int x1 = clamp_coord_int(track.x1, img.cols - 1);
        int y1 = clamp_coord_int(track.y1, img.rows - 1);
        int x2 = clamp_coord_int(track.x2, img.cols - 1);
        int y2 = clamp_coord_int(track.y2, img.rows - 1);
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Scalar color = tracker_color_from_id(track.track_id);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thickness);

        const char* label = (track.label >= 0 && track.label < (int)coco_labels.size())
                            ? coco_labels[track.label].c_str() : "unknown";
        snprintf(text, sizeof(text), "ID:%d %s %.1f%%", track.track_id, label, track.score * 100.0f);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseLine);
        int tx = x1;
        int ty = y1 - label_size.height - baseLine;
        if (ty < 0) ty = 0;
        if (tx + label_size.width > img.cols) tx = img.cols - label_size.width;
        cv::rectangle(img, cv::Rect(cv::Point(tx, ty),
                      cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);
        cv::putText(img, text, cv::Point(tx, ty + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);

        const auto& traj = track.trajectory;
        int start_idx = (traj.size() > 20) ? ((int)traj.size() - 20) : 0;
        for (size_t i = (size_t)start_idx + 1; i < traj.size(); ++i) {
            int px1 = clamp_coord_int(traj[i - 1].first, img.cols - 1);
            int py1 = clamp_coord_int(traj[i - 1].second, img.rows - 1);
            int px2 = clamp_coord_int(traj[i].first, img.cols - 1);
            int py2 = clamp_coord_int(traj[i].second, img.rows - 1);
            cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2), color, trail_thickness);
        }
        drawn++;
    }
    return drawn;
}

int rknn_lite::draw_unmatched_detections(cv::Mat& img, const object_detect_result_list& od_results,
                                         const std::vector<TrackerResultItem>& tracks) {
    char text[256];
    int drawn = 0;
    for (int i = 0; i < od_results.count; ++i) {
        const object_detect_result* det = &(od_results.results[i]);
        float det_x1 = det->box.left;
        float det_y1 = det->box.top;
        float det_x2 = det->box.right;
        float det_y2 = det->box.bottom;

        bool matched = false;
        for (const auto& track : tracks) {
            if (!track.active || track.track_id < 0) continue;
            float xx1 = std::max(det_x1, track.x1);
            float yy1 = std::max(det_y1, track.y1);
            float xx2 = std::min(det_x2, track.x2);
            float yy2 = std::min(det_y2, track.y2);
            float inter_w = std::max(0.0f, xx2 - xx1);
            float inter_h = std::max(0.0f, yy2 - yy1);
            float inter = inter_w * inter_h;
            float area1 = std::max(0.0f, det_x2 - det_x1) * std::max(0.0f, det_y2 - det_y1);
            float area2 = std::max(0.0f, track.x2 - track.x1) * std::max(0.0f, track.y2 - track.y1);
            float uni = area1 + area2 - inter;
            float iou = (uni > 0.0f) ? (inter / uni) : 0.0f;
            if (iou > 0.30f) {
                matched = true;
                break;
            }
        }
        if (matched) continue;

        int x1 = clamp_coord_int(det_x1, img.cols - 1);
        int y1 = clamp_coord_int(det_y1, img.rows - 1);
        int x2 = clamp_coord_int(det_x2, img.cols - 1);
        int y2 = clamp_coord_int(det_y2, img.rows - 1);
        if (x2 <= x1 || y2 <= y1) continue;

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        const char* label = (det->cls_id >= 0 && det->cls_id < (int)coco_labels.size())
                            ? coco_labels[det->cls_id].c_str() : "unknown";
        snprintf(text, sizeof(text), "%s %.1f%%", label, det->prop * 100.0f);
        cv::putText(img, text, cv::Point(x1, std::max(0, y1 - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        drawn++;
    }
    return drawn;
}

std::vector<TrackerResultItem> rknn_lite::update_bytetrack(const object_detect_result_list& od_results) {
    last_tracker_feature_match_ = true;
    std::vector<Object> objects;
    for (int i = 0; i < od_results.count; i++) {
        const object_detect_result* det = &(od_results.results[i]);
        Object obj;
        obj.rect = cv::Rect_<float>(
            det->box.left,
            det->box.top,
            det->box.right - det->box.left,
            det->box.bottom - det->box.top
        );
        obj.label = det->cls_id;
        obj.prob = det->prop;
        obj.name = "";
            objects.push_back(obj);
    }

    std::vector<TrackerResultItem> results;
    BYTETracker* shared_tracker = ensure_shared_bytetrack_tracker();
    if (shared_tracker == nullptr) {
        return results;
    }

    int stream_id = tracker_stream_id_;
    if (stream_id < 0) stream_id = 0;
    if (stream_id > 2) stream_id = 2;
    long long frame_index = ++g_shared_bytetrack_frame_counter[stream_id];

    std::vector<STrack> stracks;
    {
        std::lock_guard<std::mutex> tracker_lock(g_shared_bytetrack_runtime_mutex[stream_id]);
        stracks = shared_tracker->update(objects, 30, frame_index);
    }
    results.reserve(stracks.size());
    for (const auto& st : stracks) {
        if (!st.is_activated || st.track_id < 0) continue;
        TrackerResultItem item;
        item.track_id = st.track_id;
        item.label = st.label;
        item.score = st.score;
        item.x1 = st.tlbr[0];
        item.y1 = st.tlbr[1];
        item.x2 = st.tlbr[2];
        item.y2 = st.tlbr[3];
        item.active = st.is_activated;
        item.trajectory.assign(st.get_trajectory().begin(), st.get_trajectory().end());
        results.push_back(item);
    }
    return results;
}

std::vector<TrackerResultItem> rknn_lite::update_deepsort(cv::Mat& img, const object_detect_result_list& od_results) {
    static std::atomic<int> deepsort_debug_counter(0);
    std::vector<TrackerResultItem> results;
    if (od_results.count <= 0) {
        last_tracker_feature_match_ = true;
        return results;
    }

    std::vector<DetectBox> detections;
    detections.reserve(od_results.count);
    for (int i = 0; i < od_results.count; i++) {
        const object_detect_result* det = &(od_results.results[i]);
        DetectBox box;
        box.x1 = det->box.left;
        box.y1 = det->box.top;
        box.x2 = det->box.right;
        box.y2 = det->box.bottom;
        box.confidence = det->prop;
        box.classID = (float)det->cls_id;
        box.trackID = -1.0f;
        detections.push_back(box);
    }
    std::vector<DetectBox> raw_detections = detections;

    DeepSort* shared_tracker = ensure_shared_deepsort_tracker();
    if (shared_tracker == nullptr) {
        return results;
    }

    std::vector<Track> tracks;
    int skip_frames = get_deepsort_skip_frames();
    long long frame_index = ++g_shared_deepsort_frame_counter[tracker_stream_id_];
    bool do_feature_match = (skip_frames <= 0) || (((frame_index - 1) % (skip_frames + 1)) == 0);
    last_tracker_feature_match_ = do_feature_match;
    {
        std::lock_guard<std::mutex> tracker_lock(g_shared_deepsort_runtime_mutex[tracker_stream_id_]);
        if (do_feature_match) {
            shared_tracker->sort(img, detections);
            g_shared_deepsort_last_detections[tracker_stream_id_] = detections;
        } else {
            const std::vector<DetectBox>& prev_detections = g_shared_deepsort_last_detections[tracker_stream_id_];
            detections = raw_detections;
            for (auto& det : detections) {
                det.trackID = -1.0f;
            }
            for (auto& cur : detections) {
                float best_iou = 0.25f;
                int matched_id = -1;
                for (const auto& prev : prev_detections) {
                    float xx1 = std::max(cur.x1, prev.x1);
                    float yy1 = std::max(cur.y1, prev.y1);
                    float xx2 = std::min(cur.x2, prev.x2);
                    float yy2 = std::min(cur.y2, prev.y2);
                    float inter_w = std::max(0.0f, xx2 - xx1);
                    float inter_h = std::max(0.0f, yy2 - yy1);
                    float inter = inter_w * inter_h;
                    float area1 = std::max(0.0f, cur.x2 - cur.x1) * std::max(0.0f, cur.y2 - cur.y1);
                    float area2 = std::max(0.0f, prev.x2 - prev.x1) * std::max(0.0f, prev.y2 - prev.y1);
                    float uni = area1 + area2 - inter;
                    float iou = (uni > 0.0f) ? (inter / uni) : 0.0f;
                    if (iou > best_iou) {
                        best_iou = iou;
                        matched_id = (int)prev.trackID;
                    }
                }
                if (matched_id >= 0) {
                    cur.trackID = (float)matched_id;
                }
            }
        }
        tracks = shared_tracker->get_confirmed_tracks();
    }
    int stale_tracks = 0;
    bool force_debug = false;
    const char* env_debug = getenv("DEEPSORT_DEBUG");
    if (env_debug && (*env_debug == '1' || *env_debug == 'y' || *env_debug == 'Y')) {
        force_debug = true;
    }
    int dbg_idx = deepsort_debug_counter.fetch_add(1) + 1;
    bool print_debug = force_debug || (dbg_idx % 120 == 0);
    results.reserve(detections.size());
    if (print_debug) {
        printf("[DeepSORT dbg] input det=%zu frame=%lld skip=%d feature_match=%d\n",
               detections.size(), frame_index, skip_frames, do_feature_match ? 1 : 0);
        for (size_t i = 0; i < detections.size() && i < 6; ++i) {
            const auto& det = detections[i];
            printf("  det[%zu] cls=%d conf=%.2f box=(%.1f,%.1f,%.1f,%.1f)\n",
                   i, (int)det.classID, det.confidence, det.x1, det.y1, det.x2, det.y2);
        }
    }

    std::map<int, const Track*> confirmed_track_map;
    for (const auto& track : tracks) {
        if (track.time_since_update > 1) {
            stale_tracks++;
            if (print_debug && stale_tracks <= 6) {
                DETECTBOX stale_tlwh = track.to_tlwh();
                printf("  stale track id=%d cls=%d conf=%.2f hits=%d age=%d tsu=%d box=(%.1f,%.1f,%.1f,%.1f)\n",
                       track.track_id, track.cls, track.conf, track.hits, track.age, track.time_since_update,
                       stale_tlwh(0), stale_tlwh(1), stale_tlwh(0) + stale_tlwh(2), stale_tlwh(1) + stale_tlwh(3));
            }
            continue;
        }
        confirmed_track_map[track.track_id] = &track;
    }

    for (const auto& det : detections) {
        int track_id = (int)det.trackID;
        if (track_id < 0) continue;
        TrackerResultItem item;
        item.track_id = track_id;
        item.label = (int)det.classID;
        item.score = det.confidence;
        item.x1 = det.x1;
        item.y1 = det.y1;
        item.x2 = det.x2;
        item.y2 = det.y2;
        item.active = true;
        auto it = confirmed_track_map.find(track_id);
        if (it != confirmed_track_map.end()) {
            item.label = it->second->cls >= 0 ? it->second->cls : item.label;
            item.score = it->second->conf >= 0.0f ? it->second->conf : item.score;
            item.trajectory = it->second->get_trajectory();
        }
        results.push_back(item);
        if (print_debug && results.size() <= 6) {
            printf("  keep  track id=%d cls=%d conf=%.2f hits=%d age=%d tsu=%d box=(%.1f,%.1f,%.1f,%.1f)\n",
                   item.track_id, item.label, item.score,
                   it != confirmed_track_map.end() ? it->second->hits : -1,
                   it != confirmed_track_map.end() ? it->second->age : -1,
                   it != confirmed_track_map.end() ? it->second->time_since_update : 0,
                   item.x1, item.y1, item.x2, item.y2);
        }
    }
    if (print_debug) {
        printf("[DeepSORT dbg] confirmed=%zu fresh=%zu stale=%d\n",
               tracks.size(), results.size(), stale_tracks);
    }
    return results;
}

// 推理接口函数
int rknn_lite::interf() {
    fps_frame_count++;
    double current_time = (double)cv::getTickCount() / cv::getTickFrequency();
    if (fps_last_time == 0.0) fps_last_time = current_time;
    double elapsed = current_time - fps_last_time;
    if (elapsed >= 1.0) {
        current_fps = fps_frame_count / elapsed;
        fps_frame_count = 0;
        fps_last_time = current_time;
    }

    cv::Mat img;
    bool input_ready = false;
    if (video_dmabuf_frame_.valid) {
        input_ready = preprocess_video_dmabuf_to_input_mem();
        if (!input_ready) {
            int fallback_idx = g_preprocess_video_iomem_fallback_counter.fetch_add(1) + 1;
            if (fallback_idx <= 10 || (fallback_idx % 120) == 0) {
                printf("[Preprocess] video drm_fd -> input_mem fallback to rgb_mat path\n");
            }
        }
        clear_video_dmabuf_frame();
    }
    if (!input_ready && video_nv12_valid_) {
        input_ready = preprocess_video_nv12_to_input_mem();
    }
    if (!input_ready) {
        if (RGA_bgr_to_rgb(ori_img, img) != 0) {
            printf("RGA BGR转RGB失败，回退到OpenCV\n");
            return -1;
        }
        input_ready = input_mem_ready_ ? upload_rgb_to_input_mem(img) : false;
    }
    if (!input_ready && input_mem_ready_) {
        printf("input_mem 预处理失败\n");
        return -1;
    }
    if (!input_mem_ready_) {
        int img_width = ori_img.cols;
        int img_height = ori_img.rows;

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;

        if (img.empty()) {
            if (RGA_bgr_to_rgb(ori_img, img) != 0) {
                printf("RGA BGR转RGB失败，回退到OpenCV\n");
                return -1;
            }
        }

        cv::Mat resized_img;
        void* buf = nullptr;
        if (img_width != app_ctx.model_width || img_height != app_ctx.model_height) {
            if (RGA_resize(img, resized_img, app_ctx.model_width, app_ctx.model_height) != 0) {
                printf("RGA缩放失败，回退到OpenCV\n");
                cv::resize(img, resized_img, cv::Size(app_ctx.model_width, app_ctx.model_height));
            }
            buf = (void*)resized_img.data;
        } else {
            buf = (void*)img.data;
        }
        inputs[0].buf = buf;

        ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
        if (ret < 0) {
            printf("rknn_inputs_set 错误 ret=%d\n", ret);
            return -1;
        }
        int log_idx = g_preprocess_legacy_log_counter.fetch_add(1) + 1;
        if ((log_idx % 120) == 0) {
            printf("[Preprocess] legacy rknn_inputs_set fallback (%dx%d -> %dx%d)\n",
                   img_width, img_height, app_ctx.model_width, app_ctx.model_height);
        }
    }

    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx.is_quant);
    }

    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    ret = rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, NULL);

    float scale_w = (float)app_ctx.model_width / ori_img.cols;
    float scale_h = (float)app_ctx.model_height / ori_img.rows;

    object_detect_result_list od_results;
    post_process(&app_ctx, outputs, detection_threshold.load(), NMS_THRESH, scale_w, scale_h, &od_results);
    last_detection_count = od_results.count;

    last_tracks_.clear();
    int drawn_tracker_boxes = 0;
    if (enable_tracker_) {
        if (tracker_backend_ == TrackerBackend::DeepSort) {
            last_tracks_ = update_deepsort(img, od_results);
        } else if (tracker_backend_ == TrackerBackend::ByteTrack) {
            last_tracks_ = update_bytetrack(od_results);
        }
        drawn_tracker_boxes = draw_tracker_results(ori_img, last_tracks_);
        if (tracker_backend_ != TrackerBackend::DeepSort || !last_tracker_feature_match_) {
            drawn_tracker_boxes += draw_unmatched_detections(ori_img, od_results, last_tracks_);
        }
    }

    if (drawn_tracker_boxes == 0) {
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(ori_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0));

            const char* label = (det_result->cls_id >= 0 && det_result->cls_id < (int)coco_labels.size())
                                ? coco_labels[det_result->cls_id].c_str() : "unknown";
            sprintf(text, "%s %.1f%%", label, det_result->prop * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int tx = x1;
            int ty = y1 - label_size.height - baseLine;
            if (ty < 0) ty = 0;
            if (tx + label_size.width > ori_img.cols) tx = ori_img.cols - label_size.width;

            cv::rectangle(ori_img, cv::Rect(cv::Point(tx, ty),
                          cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(ori_img, text, cv::Point(tx, ty + label_size.height),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }

    char fps_text[64];
    sprintf(fps_text, "FPS: %.1f", current_fps);
    cv::rectangle(ori_img, cv::Point(5, 5), cv::Point(120, 30), cv::Scalar(0, 0, 0), -1);
    cv::putText(ori_img, fps_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    ret = rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);
    return 0;
}

// 设置是否使用跟踪器
void rknn_lite::set_use_tracker(bool use) {
    use_tracker_this_frame_ = use && enable_tracker_;
}

// 设置检测置信度阈值
void rknn_lite::set_detection_threshold(float threshold) {
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    detection_threshold.store(threshold);
    printf("[Threshold] 置信度阈值已设置为: %.2f\n", threshold);
}

// 获取检测置信度阈值
float rknn_lite::get_detection_threshold() {
    return detection_threshold.load();
}

// 设置标签文件覆盖路径（空字符串表示使用默认标签）
void rknn_lite::set_label_file_override(const std::string& label_file) {
    std::lock_guard<std::mutex> lock(g_label_file_mutex);
    g_label_file_override = label_file;
    g_label_cache_key.clear();
}

// 获取标签文件覆盖路径
std::string rknn_lite::get_label_file_override() {
    std::lock_guard<std::mutex> lock(g_label_file_mutex);
    return g_label_file_override;
}

bool rknn_lite::set_tracker_backend(const std::string& backend_name) {
    TrackerBackend backend;
    if (!parse_tracker_backend_name(backend_name, &backend)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
        g_tracker_backend_override = backend;
    }
    if (backend != TrackerBackend::DeepSort) {
        reset_shared_deepsort_trackers();
    }
    if (backend != TrackerBackend::ByteTrack) {
        reset_shared_bytetrack_trackers();
    }
    return true;
}

std::string rknn_lite::get_tracker_backend_name() {
    std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
    return tracker_backend_to_string(g_tracker_backend_override);
}

void rknn_lite::set_tracker_reid_model_override(const std::string& model_file) {
    {
        std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
        g_tracker_reid_model_override = model_file;
    }
    reset_shared_deepsort_trackers();
}

std::string rknn_lite::get_tracker_reid_model_override() {
    std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
    return g_tracker_reid_model_override;
}

std::string rknn_lite::resolve_tracker_reid_model() {
    std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
    return resolve_reid_model_path_locked();
}

void rknn_lite::set_deepsort_skip_frames(int skip_frames) {
    std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
    g_deepsort_skip_frames = sanitize_deepsort_skip_frames(skip_frames);
}

int rknn_lite::get_deepsort_skip_frames() {
    std::lock_guard<std::mutex> lock(g_tracker_config_mutex);
    return g_deepsort_skip_frames;
}

void rknn_lite::reset_shared_deepsort_trackers() {
    std::lock_guard<std::mutex> lock(g_shared_deepsort_init_mutex);
    for (DeepSort*& tracker : g_shared_deepsort_trackers) {
        if (tracker != nullptr) {
            delete tracker;
            tracker = nullptr;
        }
    }
    g_shared_deepsort_model_path.clear();
    for (int i = 0; i < 3; ++i) {
        g_shared_deepsort_frame_counter[i].store(0);
        g_shared_deepsort_last_detections[i].clear();
    }
}

void rknn_lite::reset_shared_bytetrack_trackers() {
    std::lock_guard<std::mutex> lock(g_shared_bytetrack_init_mutex);
    for (BYTETracker*& tracker : g_shared_bytetrack_trackers) {
        if (tracker != nullptr) {
            delete tracker;
            tracker = nullptr;
        }
    }
    for (int i = 0; i < 3; ++i) {
        g_shared_bytetrack_frame_counter[i].store(0);
    }
}

void rknn_lite::set_tracker_stream_id(int stream_id) {
    if (stream_id < 0) stream_id = 0;
    if (stream_id > 2) stream_id = 2;
    tracker_stream_id_ = stream_id;
}

BYTETracker* rknn_lite::ensure_shared_bytetrack_tracker() {
    if (tracker_backend_ != TrackerBackend::ByteTrack) {
        return nullptr;
    }

    int stream_id = tracker_stream_id_;
    if (stream_id < 0) stream_id = 0;
    if (stream_id > 2) stream_id = 2;

    std::lock_guard<std::mutex> lock(g_shared_bytetrack_init_mutex);
    if (g_shared_bytetrack_trackers[stream_id] == nullptr) {
        g_shared_bytetrack_trackers[stream_id] = new BYTETracker(30, 30, 0.5, 0.6, 0.8);
        printf("ByteTrack 共享实例已创建: stream=%d\n", stream_id);
    }
    return g_shared_bytetrack_trackers[stream_id];
}

DeepSort* rknn_lite::ensure_shared_deepsort_tracker() {
    if (tracker_backend_ != TrackerBackend::DeepSort || reid_model_path_.empty() || !is_readable_file(reid_model_path_)) {
        return nullptr;
    }

    int stream_id = tracker_stream_id_;
    if (stream_id < 0) stream_id = 0;
    if (stream_id > 2) stream_id = 2;

    std::lock_guard<std::mutex> lock(g_shared_deepsort_init_mutex);
    if (!g_shared_deepsort_model_path.empty() && g_shared_deepsort_model_path != reid_model_path_) {
        for (DeepSort*& tracker : g_shared_deepsort_trackers) {
            if (tracker != nullptr) {
                delete tracker;
                tracker = nullptr;
            }
        }
        g_shared_deepsort_model_path.clear();
    }

    if (g_shared_deepsort_trackers[stream_id] == nullptr) {
        g_shared_deepsort_trackers[stream_id] = new DeepSort(reid_model_path_, 1, 512, 6, RKNN_NPU_CORE_2);
        g_shared_deepsort_model_path = reid_model_path_;
        printf("DeepSORT 共享实例已创建: stream=%d, ReID=%s\n", stream_id, reid_model_path_.c_str());
    }

    return g_shared_deepsort_trackers[stream_id];
}

// 仅检测模式：不绘制结果，只返回检测结果
int rknn_lite::interf_detect_only() {
    static std::atomic<int> infer_debug_counter(0);

    cv::Mat img;
    bool input_ready = false;
    if (video_dmabuf_frame_.valid) {
        input_ready = preprocess_video_dmabuf_to_input_mem();
        if (!input_ready) {
            int fallback_idx = g_preprocess_video_iomem_fallback_counter.fetch_add(1) + 1;
            if (fallback_idx <= 10 || (fallback_idx % 120) == 0) {
                printf("[Preprocess] video drm_fd -> input_mem fallback to rgb_mat path\n");
            }
        }
        clear_video_dmabuf_frame();
    }
    if (!input_ready && video_nv12_valid_) {
        input_ready = preprocess_video_nv12_to_input_mem();
    }
    if (!input_ready) {
        if (RGA_bgr_to_rgb(ori_img, img) != 0) {
            printf("RGA BGR转RGB失败\n");
            return -1;
        }
        input_ready = input_mem_ready_ ? upload_rgb_to_input_mem(img) : false;
    }
    if (!input_ready && input_mem_ready_) return -1;

    if (!input_mem_ready_) {
        int img_width = img.cols;
        int img_height = img.rows;

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;

        cv::Mat resized_img;
        if (img_width != app_ctx.model_width || img_height != app_ctx.model_height) {
            if (RGA_resize(img, resized_img, app_ctx.model_width, app_ctx.model_height) != 0) {
                cv::resize(img, resized_img, cv::Size(app_ctx.model_width, app_ctx.model_height));
            }
            inputs[0].buf = resized_img.data;
        } else {
            inputs[0].buf = img.data;
        }

        ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
        if (ret < 0) return -1;
        int log_idx = g_preprocess_legacy_log_counter.fetch_add(1) + 1;
        if ((log_idx % 120) == 0) {
            printf("[Preprocess] legacy rknn_inputs_set fallback (%dx%d -> %dx%d)\n",
                   img_width, img_height, app_ctx.model_width, app_ctx.model_height);
        }
    }
    
    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    if (ret < 0) return -1;
    
    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx.is_quant);
    }
    rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, nullptr);
    
    float scale_w = (float)app_ctx.model_width / ori_img.cols;
    float scale_h = (float)app_ctx.model_height / ori_img.rows;
    
    object_detect_result_list od_results;
    post_process(&app_ctx, outputs, detection_threshold.load(), NMS_THRESH, scale_w, scale_h, &od_results);
    last_detection_count = od_results.count;

    last_tracks_.clear();
    int drawn_tracker_boxes = 0;
    int drawn_det_boxes = 0;

    if (use_tracker_this_frame_) {
        if (tracker_backend_ == TrackerBackend::DeepSort) {
            last_tracks_ = update_deepsort(img, od_results);
        } else if (tracker_backend_ == TrackerBackend::ByteTrack) {
            last_tracks_ = update_bytetrack(od_results);
        }
        drawn_tracker_boxes = draw_tracker_results(ori_img, last_tracks_);
        if (tracker_backend_ != TrackerBackend::DeepSort || !last_tracker_feature_match_) {
            drawn_tracker_boxes += draw_unmatched_detections(ori_img, od_results, last_tracks_);
        }
    }

    if (drawn_tracker_boxes == 0) {
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det = &(od_results.results[i]);
            int x1 = det->box.left, y1 = det->box.top;
            int x2 = det->box.right, y2 = det->box.bottom;
            cv::rectangle(ori_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            drawn_det_boxes++;
            const char* label = (det->cls_id >= 0 && det->cls_id < (int)coco_labels.size())
                                ? coco_labels[det->cls_id].c_str() : "unknown";
            char text[256];
            snprintf(text, sizeof(text), "%s %.1f%%", label, det->prop * 100);
            cv::putText(ori_img, text, cv::Point(x1, y1 - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

    int dbg = infer_debug_counter.fetch_add(1) + 1;
    if ((dbg % 180) == 0) {
        printf("[Draw infer] det=%d, draw_det=%d, draw_track=%d, backend=%s, use_tracker=%d\n",
               od_results.count, drawn_det_boxes, drawn_tracker_boxes,
               tracker_backend_to_string(tracker_backend_).c_str(), use_tracker_this_frame_ ? 1 : 0);
    }
    
    rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);
    return 0;
}

#endif // _RKNNPOOL_HPP
