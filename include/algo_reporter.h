#pragma once

#include <opencv2/core/core.hpp>
#include <string>

// C++ 端：只负责在命中告警时打印/记录事件，不做 HTTP，上报交给外部脚本或服务处理
// 仅用于：摔倒/打架/持刀 三类告警（cls_id: 1/2/3）
namespace algo_reporter {

struct Config {
  // 触发阈值与限频（由环境变量 CV_MIN_CONF_* / CV_COOLDOWN_MS 控制）
  float min_conf_fall = 0.50f;
  float min_conf_fight = 0.50f;
  float min_conf_knife = 0.50f;
  int cooldown_ms = 3000; // 同类事件最小上报间隔
};

// 从环境变量读取配置（未设置则用默认值）
Config load_config_from_env();

// 以下两个接口为了兼容原调用点，现均为空实现
bool ensure_login(const Config& cfg);
void start_heartbeat(const Config& cfg);

// cls_id: 0 行人 1 摔倒 2 打架 3 持刀
// conf: det_result->prop
// frame_bgr: 当前帧（可带框或原图）
void maybe_report_event(const Config& cfg, int cls_id, float conf, const cv::Mat& frame_bgr);

}  // namespace algo_reporter
