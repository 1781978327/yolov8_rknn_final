#ifndef _RKNNPOOL_HPP
#define _RKNNPOOL_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "postprocess.h"
#include "rk_common.h"
#include "rknn_api.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "BYTETracker.h"

// 动态标签名称，从txt文件加载
static std::vector<std::string> coco_labels;
static std::string g_label_file_override;
static std::mutex g_label_file_mutex;
static std::string g_label_cache_key;

// FPS 计算变量
static int fps_frame_count = 0;
static double fps_last_time = 0.0;
static double current_fps = 0.0;

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
    rknn_app_context_t app_ctx;  // RKNN应用上下文
    int ret;  // 函数返回值
    BYTETracker* tracker_;  // BYTETracker 跟踪器
    long long frame_count_;  // 帧计数器
    bool enable_tracker_;  // 是否启用跟踪
    bool use_tracker_this_frame_;  // 本帧是否使用跟踪器
    static float detection_threshold;  // 置信度阈值

public:
    cv::Mat ori_img;  // 原始图像
    static std::atomic<int> last_detection_count;  // 最后检测到的目标数量

    rknn_lite(char* model_path, int core_id);
    ~rknn_lite();
    int interf();
    int interf_detect_only();  // 仅推理，不绘制
    void set_use_tracker(bool use);  // 设置是否使用跟踪器
    bool is_tracker_enabled() const { return enable_tracker_; }
    std::vector<STrack> get_last_tracks() const { return last_tracks_; }
    static void set_detection_threshold(float threshold);  // 设置置信度阈值
    static float get_detection_threshold();  // 获取置信度阈值
    static void set_label_file_override(const std::string& label_file);  // 设置标签txt覆盖路径，空字符串=默认
    static std::string get_label_file_override();  // 获取标签txt覆盖路径

    int RGA_bgr_to_rgb(const cv::Mat& bgr_image, cv::Mat &rgb_image);
    int RGA_resize(const cv::Mat& src, cv::Mat& dst, int dst_width, int dst_height);

private:
    std::vector<STrack> last_tracks_;  // 最后一帧的跟踪结果
};

// 静态成员初始化
float rknn_lite::detection_threshold = 0.5f;
std::atomic<int> rknn_lite::last_detection_count(0);

// 构造函数：初始化RKNN模型
rknn_lite::rknn_lite(char* model_path, int core_id) {
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));
    tracker_ = nullptr;
    frame_count_ = 0;
    use_tracker_this_frame_ = false;

    // 总是启用跟踪器（运行时可通过 API 控制开关）
    enable_tracker_ = true;

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

    // 初始化 BYTETracker（如果启用）
    if (enable_tracker_) {
        tracker_ = new BYTETracker(30, 30, 0.5, 0.6, 0.8);
        
        // 从环境变量读取轨迹时长（秒），转换为帧数
        const char* trail_sec_env = getenv("TRACK_TRAIL_SECONDS");
        int trail_seconds = 3;  // 默认1秒
        if (trail_sec_env) {
            trail_seconds = atoi(trail_sec_env);
            if (trail_seconds <= 0) trail_seconds = 1;
        }
        // 假设25fps，计算对应的帧数
        int trail_frames = trail_seconds * 25;
        STrack::set_max_trail_length(trail_frames);
        
        printf("BYTETracker 跟踪已启用，轨迹保留 %d 秒（约 %d 帧）\n", trail_seconds, trail_frames);
    }

    // 设置NPU核心掩码
    rknn_core_mask core_mask;
    switch(core_id % 3) {
        case 0: core_mask = RKNN_NPU_CORE_0; break;
        case 1: core_mask = RKNN_NPU_CORE_1; break;
        default: core_mask = RKNN_NPU_CORE_2;
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
}

// 析构函数：释放资源
rknn_lite::~rknn_lite() {
    if (app_ctx.rknn_ctx != 0) {
        rknn_destroy(app_ctx.rknn_ctx);
    }
    if (app_ctx.input_attrs != nullptr) {
        delete[] app_ctx.input_attrs;
    }
    if (app_ctx.output_attrs != nullptr) {
        delete[] app_ctx.output_attrs;
    }
    if (tracker_ != nullptr) {
        delete tracker_;
        tracker_ = nullptr;
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

// 推理接口函数
int rknn_lite::interf() {
    // FPS 计算
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
    // 使用RGA进行BGR到RGB转换
    if (RGA_bgr_to_rgb(ori_img, img) != 0) {
        printf("RGA BGR转RGB失败，回退到OpenCV\n");
        return -1;
    }
    
    int img_width = img.cols;
    int img_height = img.rows;
    
    // 准备输入张量
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx.model_width * app_ctx.model_height * app_ctx.model_channel;
    
    // 使用RGA进行图像缩放
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
    
    // 设置输入
    ret = rknn_inputs_set(app_ctx.rknn_ctx, app_ctx.io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set 错误 ret=%d\n", ret);
        return -1;
    }
    
    // 准备输出
    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx.is_quant);
    }
    
    // 执行推理
    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    ret = rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, NULL);
    
    // 后处理
    float scale_w = (float)app_ctx.model_width / img_width;
    float scale_h = (float)app_ctx.model_height / img_height;
    
    object_detect_result_list od_results;
    post_process(&app_ctx, outputs, detection_threshold, NMS_THRESH, scale_w, scale_h, &od_results);
    
    // 更新检测数量
    last_detection_count = od_results.count;
    
    // 绘制检测结果
    char text[256];
    
    // 如果启用跟踪，使用 BYTETracker
    if (enable_tracker_ && tracker_ != nullptr) {
        // 准备跟踪输入
        std::vector<Object> objects;
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det = &(od_results.results[i]);
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
        
        // 更新跟踪器
        frame_count_++;
        std::vector<STrack> stracks = tracker_->update(objects, 30, frame_count_);
        
        // 绘制跟踪结果
        for (const auto& st : stracks) {
            if (!st.is_activated) continue;
            
            int x1 = (int)st.tlbr[0];
            int y1 = (int)st.tlbr[1];
            int x2 = (int)st.tlbr[2];
            int y2 = (int)st.tlbr[3];
            
            // 获取跟踪颜色
            cv::Scalar color = tracker_->get_color(st.track_id);
            
            // 绘制跟踪框
            cv::rectangle(ori_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            
            // 绘制标签（包含跟踪ID）
            const char* label = (st.label >= 0 && st.label < (int)coco_labels.size())
                                ? coco_labels[st.label].c_str() : "unknown";
            sprintf(text, "ID:%d %s %.1f%%", st.track_id, label, st.score * 100);
            
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            int x = x1;
            int y = y1 - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > ori_img.cols) x = ori_img.cols - label_size.width;
            
            // 绘制背景
            cv::rectangle(ori_img, cv::Rect(cv::Point(x, y), 
                          cv::Size(label_size.width, label_size.height + baseLine)), 
                          color, -1);
            
            // 绘制文字
            cv::putText(ori_img, text, cv::Point(x, y + label_size.height), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            
            // 绘制轨迹
            const auto& traj = st.get_trajectory();
            for (size_t j = 1; j < traj.size(); j++) {
                cv::line(ori_img, 
                         cv::Point(traj[j-1].first, traj[j-1].second),
                         cv::Point(traj[j].first, traj[j].second),
                         color, 1);
            }
        }
    } else {
        // 不启用跟踪，原始绘制方式
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
            
            int x = x1;
            int y = y1 - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > ori_img.cols) x = ori_img.cols - label_size.width;
            
            cv::rectangle(ori_img, cv::Rect(cv::Point(x, y), 
                          cv::Size(label_size.width, label_size.height + baseLine)), 
                          cv::Scalar(255, 255, 255), -1);
            
            cv::putText(ori_img, text, cv::Point(x, y + label_size.height), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    
    // 在画面左上角显示 FPS
    char fps_text[64];
    sprintf(fps_text, "FPS: %.1f", current_fps);
    cv::rectangle(ori_img, cv::Point(5, 5), cv::Point(120, 30), cv::Scalar(0, 0, 0), -1);
    cv::putText(ori_img, fps_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // 释放输出
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
    detection_threshold = threshold;
    printf("[Threshold] 置信度阈值已设置为: %.2f\n", threshold);
}

// 获取检测置信度阈值
float rknn_lite::get_detection_threshold() {
    return detection_threshold;
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

// 仅检测模式：不绘制结果，只返回检测结果
int rknn_lite::interf_detect_only() {
    static std::atomic<int> infer_debug_counter(0);

    cv::Mat img;
    if (RGA_bgr_to_rgb(ori_img, img) != 0) {
        printf("RGA BGR转RGB失败\n");
        return -1;
    }
    
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
    
    ret = rknn_run(app_ctx.rknn_ctx, nullptr);
    if (ret < 0) return -1;
    
    rknn_output outputs[app_ctx.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < app_ctx.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx.is_quant);
    }
    rknn_outputs_get(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs, nullptr);
    
    float scale_w = (float)app_ctx.model_width / img_width;
    float scale_h = (float)app_ctx.model_height / img_height;
    
    object_detect_result_list od_results;
    post_process(&app_ctx, outputs, detection_threshold, NMS_THRESH, scale_w, scale_h, &od_results);
    
    // 更新检测数量
    last_detection_count = od_results.count;
    
    last_tracks_.clear();
    int drawn_tracker_boxes = 0;
    int drawn_det_boxes = 0;
    
    // 如果启用跟踪，更新跟踪器
    if (use_tracker_this_frame_ && tracker_ != nullptr) {
        std::vector<Object> objects;
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det = &(od_results.results[i]);
            Object obj;
            obj.rect = cv::Rect_<float>(det->box.left, det->box.top,
                det->box.right - det->box.left, det->box.bottom - det->box.top);
            obj.label = det->cls_id;
            obj.prob = det->prop;
            objects.push_back(obj);
        }
        frame_count_++;
        last_tracks_ = tracker_->update(objects, 30, frame_count_);
        
        // 绘制跟踪结果
        for (const auto& st : last_tracks_) {
            if (!st.is_activated) continue;
            int x1 = (int)st.tlbr[0], y1 = (int)st.tlbr[1];
            int x2 = (int)st.tlbr[2], y2 = (int)st.tlbr[3];
            if (x1 > x2) std::swap(x1, x2);
            if (y1 > y2) std::swap(y1, y2);
            x1 = std::max(0, std::min(x1, ori_img.cols - 1));
            y1 = std::max(0, std::min(y1, ori_img.rows - 1));
            x2 = std::max(0, std::min(x2, ori_img.cols - 1));
            y2 = std::max(0, std::min(y2, ori_img.rows - 1));
            if (x2 <= x1 || y2 <= y1) continue;
            cv::Scalar color = tracker_->get_color(st.track_id);
            cv::rectangle(ori_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
            drawn_tracker_boxes++;
            const char* label = (st.label >= 0 && st.label < (int)coco_labels.size())
                                ? coco_labels[st.label].c_str() : "unknown";
            char text[256];
            snprintf(text, sizeof(text), "ID:%d %s %.1f%%", st.track_id, label, st.score * 100);
            cv::putText(ori_img, text, cv::Point(x1, y1 - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            // 绘制轨迹
            const auto& traj = st.get_trajectory();
            for (size_t j = 1; j < traj.size(); j++) {
                int px1 = (int)traj[j-1].first;
                int py1 = (int)traj[j-1].second;
                int px2 = (int)traj[j].first;
                int py2 = (int)traj[j].second;
                px1 = std::max(0, std::min(px1, ori_img.cols - 1));
                py1 = std::max(0, std::min(py1, ori_img.rows - 1));
                px2 = std::max(0, std::min(px2, ori_img.cols - 1));
                py2 = std::max(0, std::min(py2, ori_img.rows - 1));
                cv::line(ori_img,
                         cv::Point(px1, py1),
                         cv::Point(px2, py2),
                         color, 2);
            }
        }
    } else {
        // 不使用跟踪，只绘制检测框
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
        printf("[Draw infer] det=%d, draw_det=%d, draw_track=%d, use_tracker=%d\n",
               od_results.count, drawn_det_boxes, drawn_tracker_boxes, use_tracker_this_frame_ ? 1 : 0);
    }
    
    rknn_outputs_release(app_ctx.rknn_ctx, app_ctx.io_num.n_output, outputs);
    return 0;
}

#endif // _RKNNPOOL_HPP
