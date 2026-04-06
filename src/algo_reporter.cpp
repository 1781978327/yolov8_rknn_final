#include "algo_reporter.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <opencv2/imgcodecs.hpp>

namespace algo_reporter {
namespace {

long getenv_long(const char* k, long defv) {
  const char* v = std::getenv(k);
  if (!v || !*v) return defv;
  char* end = nullptr;
  long x = std::strtol(v, &end, 10);
  return (end && end != v) ? x : defv;
}

float getenv_float(const char* k, float defv) {
  const char* v = std::getenv(k);
  if (!v || !*v) return defv;
  char* end = nullptr;
  float x = std::strtof(v, &end);
  return (end && end != v) ? x : defv;
}

int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

struct State {
  std::mutex mu;
  std::unordered_map<int, int64_t> last_report_ms_by_cls;
};

State& st() {
  static State s;
  return s;
}

float min_conf_for_cls(const Config& cfg, int cls_id) {
  switch (cls_id) {
    case 1: return cfg.min_conf_fall;
    case 2: return cfg.min_conf_fight;
    case 3: return cfg.min_conf_knife;
    default: return 1.0f;
  }
}

}  // namespace

Config load_config_from_env() {
  Config cfg;
  cfg.min_conf_fall = getenv_float("CV_MIN_CONF_FALL", cfg.min_conf_fall);
  cfg.min_conf_fight = getenv_float("CV_MIN_CONF_FIGHT", cfg.min_conf_fight);
  cfg.min_conf_knife = getenv_float("CV_MIN_CONF_KNIFE", cfg.min_conf_knife);
  cfg.cooldown_ms = (int)getenv_long("CV_COOLDOWN_MS", cfg.cooldown_ms);
  return cfg;
}

bool ensure_login(const Config& /*cfg*/) {
  // 兼容旧调用点：不再做任何登录逻辑
  return true;
}

void start_heartbeat(const Config& /*cfg*/) {
  // 兼容旧调用点：不再发送心跳，上报由外部脚本负责
}

void maybe_report_event(const Config& cfg, int cls_id, float conf, const cv::Mat& frame_bgr) {
  if (cls_id != 1 && cls_id != 2 && cls_id != 3) return;
  if (conf < min_conf_for_cls(cfg, cls_id)) return;
  if (frame_bgr.empty()) return;

  // 限频：同一类事件 cooldown 内只打印一次
  {
    State& s = st();
    std::lock_guard<std::mutex> lk(s.mu);
    int64_t now = now_ms();
    auto it = s.last_report_ms_by_cls.find(cls_id);
    if (it != s.last_report_ms_by_cls.end() && now - it->second < cfg.cooldown_ms) {
      return;
    }
    s.last_report_ms_by_cls[cls_id] = now;
  }

  printf("[algo_reporter] 命中告警: cls=%d conf=%.3f\n", cls_id, conf);
  fflush(stdout);

  // 将当前帧保存为临时 JPEG，供独立的 report_test 进程读取并上报
  const char* img_path = "/tmp/yolo_event.jpg";
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
  try {
    cv::imwrite(img_path, frame_bgr, params);
  } catch (...) {
    // 保存失败就只打印日志，不影响主流程
    return;
  }

  // 启动独立进程进行 HTTP 上报，避免在推理进程内部直接使用 libcurl 导致卡死
  std::thread([cls_id, conf]() {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
                  "/home/orangepi/Desktop/yolov8-rk3588-cpp-3-15/build/report_test %d %.3f >/tmp/report_test.log 2>&1",
                  cls_id, conf);
    std::system(cmd);
  }).detach();
}

}  // namespace algo_reporter
