// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#include <string>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
#include "im2d.h"
#include "rga.h"
#include "RgaUtils.h"
#include "algo_reporter.h"
#ifdef USE_RTSP_MPP
#include "rtsp_mpp_sender.h"
#endif
using std::queue;
using std::time;
using std::time_t;
using std::vector;

int main(int argc, char **argv)
{
  char *model_name = NULL;
  if (argc < 2)
  {
    printf("Usage: %s <rknn model> [video_path] [output_path]\n", argv[0]);
    printf("  双摄: video_path 用逗号分隔；双路 RTSP: output_path 写两个地址逗号分隔\n");
    printf("  例: %s model.rknn /dev/video0,/dev/video2 rtsp://ip:8554/cam0,rtsp://ip:8554/cam1\n", argv[0]);
    printf("\n环境变量:\n");
    printf("  CV_API_BASE    后端 API 地址, 例如 http://127.0.0.1:8080/api\n");
    printf("  CV_USER        登录用户名, 默认 algo\n");
    printf("  CV_PASS        登录密码, 默认 algo123\n");
    printf("  CV_CAMERA_ID   摄像头 ID, 默认 1\n");
    printf("  CV_COOLDOWN_MS 同一类告警最小上报间隔, 默认 3000\n");
    printf("  CV_MIN_CONF_FALL  摔倒告警最小置信度, 默认 0.50\n");
    printf("  CV_MIN_CONF_FIGHT 打架告警最小置信度, 默认 0.50\n");
    printf("  CV_MIN_CONF_KNIFE 持刀告警最小置信度, 默认 0.50\n");
    printf("  CV_HEARTBEAT_SEC  心跳间隔秒数, 默认 10\n");
    printf("  CV_HEADLESS    1=后台无界面(imshow/窗口关闭), 0=开启预览\n");
    printf("  TRACK_TRAIL_SECONDS 轨迹保留秒数, 默认 4 (需配合 ENABLE_TRACKER=1 使用)\n");
    printf("  TRACKER_BACKEND 跟踪算法: bytetrack(默认) / deepsort\n");
    printf("  TRACKER_REID_MODEL DeepSORT ReID 模型路径，未设置时自动找 osnet_x0_25_market.rknn\n");
    return -1;
  }
  model_name = (char *)argv[1]; // 参数一，模型所在路径

  const char* env_tracker_backend = getenv("TRACKER_BACKEND");
  const char* env_tracker_reid = getenv("TRACKER_REID_MODEL");
  if (env_tracker_backend && *env_tracker_backend) {
    if (!rknn_lite::set_tracker_backend(env_tracker_backend)) {
      printf("警告: 无效的 TRACKER_BACKEND=%s，将继续使用默认 bytetrack\n", env_tracker_backend);
    }
  }
  if (env_tracker_reid && *env_tracker_reid) {
    rknn_lite::set_tracker_reid_model_override(env_tracker_reid);
  }
  printf("跟踪算法: %s | ReID: %s\n",
         rknn_lite::get_tracker_backend_name().c_str(),
         rknn_lite::resolve_tracker_reid_model().empty() ? "(none)" : rknn_lite::resolve_tracker_reid_model().c_str());

  // 启动算法上报心跳（后台线程），不会阻塞主循环
  algo_reporter::Config hb_cfg = algo_reporter::load_config_from_env();
  algo_reporter::start_heartbeat(hb_cfg);

  // 视频路径：单一源 或 双摄像头 "0,1" / "/dev/video0,/dev/video2"
  std::string video_path;
  if (argc >= 3) {
    video_path = argv[2];
  }
  std::string video_path0, video_path1;
  bool dual_cam = false;
  size_t comma = video_path.find(',');
  if (comma == std::string::npos)
    comma = video_path.find('，');  // 全角逗号
  if (comma != std::string::npos) {
    video_path0 = video_path.substr(0, comma);
    video_path1 = video_path.substr(comma + 1);
    // 去掉首尾空格
    auto trim = [](std::string& s) {
      size_t a = s.find_first_not_of(" \t");
      if (a == std::string::npos) s.clear();
      else { size_t b = s.find_last_not_of(" \t"); s = s.substr(a, b - a + 1); }
    };
    trim(video_path0);
    trim(video_path1);
    dual_cam = true;
  } else {
    video_path0 = video_path;
  }

  // 输出：单摄一个地址；双摄可写两个 RTSP 地址，逗号分隔，如 rtsp://a/cam0,rtsp://a/cam1
  std::string output_path;
  if (argc >= 4) {
    output_path = argv[3];
  }
  std::string output_path0, output_path1;
  if (dual_cam && output_path.find(',') != std::string::npos) {
    size_t co = output_path.find(',');
    output_path0 = output_path.substr(0, co);
    output_path1 = output_path.substr(co + 1);
    auto trim = [](std::string& s) {
      size_t a = s.find_first_not_of(" \t");
      if (a == std::string::npos) s.clear();
      else { size_t b = s.find_last_not_of(" \t"); s = s.substr(a, b - a + 1); }
    };
    trim(output_path0);
    trim(output_path1);
  }
  // 第五/第六个参数（可选）：overview 拼接流 RTSP 地址
  // argv[4]：原始画面拼接（不带推理框）
  // argv[5]：推理后画面拼接（带框）
  std::string overview_path_raw;
  std::string overview_path_det;
  if (argc >= 5) {
    overview_path_raw = argv[4];
  }
  if (argc >= 6) {
    overview_path_det = argv[5];
  }

#ifdef USE_RTSP_MPP
  bool use_rtsp_mpp = !dual_cam && (output_path.size() >= 7 && output_path.substr(0, 7) == "rtsp://");
  bool use_rtsp_dual = dual_cam && output_path0.size() >= 7 && output_path0.substr(0, 7) == "rtsp://" &&
                      output_path1.size() >= 7 && output_path1.substr(0, 7) == "rtsp://";
  bool use_rtsp_overview_raw = dual_cam && overview_path_raw.size() >= 7 && overview_path_raw.substr(0, 7) == "rtsp://";
  bool use_rtsp_overview_det = dual_cam && overview_path_det.size() >= 7 && overview_path_det.substr(0, 7) == "rtsp://";
#else
  bool use_rtsp_mpp = false;
  bool use_rtsp_dual = false;
  bool use_rtsp_overview_raw = false;
  bool use_rtsp_overview_det = false;
#endif

  auto open_cap = [](const std::string& path) -> cv::VideoCapture {
    cv::VideoCapture c;
    bool is_cam = path.empty() || path == "0" ||
                  (path.size() >= 10 && path.compare(0, 10, "/dev/video") == 0);
    if (is_cam) {
      if (path.empty() || path == "0")
        c.open(0, cv::CAP_V4L2);
      else
        c.open(path, cv::CAP_V4L2);
    } else {
      c.open(path, cv::CAP_FFMPEG);
    }
    return c;
  };

  cv::VideoCapture cap;
  cv::VideoCapture cap1;  // 仅双摄时使用
  if (dual_cam) {
    cap = open_cap(video_path0);
    cap1 = open_cap(video_path1);
    if (!cap.isOpened()) {
      printf("无法打开视频源 0: %s\n", video_path0.empty() ? "0" : video_path0.c_str());
      return -1;
    }
    if (!cap1.isOpened()) {
      printf("无法打开视频源 1: %s\n", video_path1.empty() ? "1" : video_path1.c_str());
      return -1;
    }
    printf("双摄模式: 源0=%s 源1=%s\n", video_path0.empty() ? "0" : video_path0.c_str(), video_path1.empty() ? "1" : video_path1.c_str());
  } else {
    cap = open_cap(video_path0);
    if (!cap.isOpened()) {
      printf("无法打开视频源: %s\n", video_path0.empty() ? "camera 0" : video_path0.c_str());
      if (video_path0.find(',') != std::string::npos || video_path0.find('，') != std::string::npos)
        printf("提示: 双摄请用英文逗号分隔两个源并重新编译 make，如: 0,1 或 /dev/video0,/dev/video2\n");
      return -1;
    }
  }

  int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  if (width <= 0 || height <= 0) {
    width = 1920;
    height = 1080;
  }
  double input_fps = cap.get(cv::CAP_PROP_FPS);
  if (input_fps <= 0 || input_fps > 120) input_fps = 25.0;
  int width1 = width, height1 = height;
  double fps1 = input_fps;
  if (dual_cam) {
    width1  = (int)cap1.get(cv::CAP_PROP_FRAME_WIDTH);
    height1 = (int)cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (width1 <= 0 || height1 <= 0) { width1 = 1920; height1 = 1080; }
    fps1 = cap1.get(cv::CAP_PROP_FPS);
    if (fps1 <= 0 || fps1 > 120) fps1 = 25.0;
  }

#ifdef USE_RTSP_MPP
  RtspMppSender* rtsp_sender = nullptr;
  RtspMppSender* rtsp_sender1 = nullptr;
  RtspMppSender* rtsp_sender_overview_raw = nullptr;
  RtspMppSender* rtsp_sender_overview_det = nullptr;
  if (use_rtsp_mpp) {
    rtsp_sender = new RtspMppSender();
    if (!rtsp_sender->init(output_path.c_str(), width, height, (int)input_fps)) {
      printf("RTSP+MPP 初始化失败: %s\n", output_path.c_str());
      delete rtsp_sender;
      rtsp_sender = nullptr;
    } else {
      printf("RTSP+MPP 硬件推流: %s, %dx%d, FPS: %.1f\n", output_path.c_str(), width, height, input_fps);
    }
  } else if (use_rtsp_dual) {
    rtsp_sender = new RtspMppSender();
    rtsp_sender1 = new RtspMppSender();
    if (!rtsp_sender->init(output_path0.c_str(), width, height, (int)input_fps)) {
      printf("RTSP+MPP Cam0 初始化失败: %s\n", output_path0.c_str());
      delete rtsp_sender; delete rtsp_sender1;
      rtsp_sender = nullptr; rtsp_sender1 = nullptr;
    } else if (!rtsp_sender1->init(output_path1.c_str(), width1, height1, (int)fps1)) {
      printf("RTSP+MPP Cam1 初始化失败: %s\n", output_path1.c_str());
      rtsp_sender->destroy(); delete rtsp_sender; delete rtsp_sender1;
      rtsp_sender = nullptr; rtsp_sender1 = nullptr;
    } else {
      printf("RTSP+MPP 双路推流: Cam0 %s %dx%d %.1ffps | Cam1 %s %dx%d %.1ffps\n",
             output_path0.c_str(), width, height, input_fps,
             output_path1.c_str(), width1, height1, fps1);
    }
    int ov_w = 1280, ov_h = 480;
    int ov_fps = (int)std::min(input_fps, fps1);
    if (use_rtsp_overview_raw) {
      rtsp_sender_overview_raw = new RtspMppSender();
      if (!rtsp_sender_overview_raw->init(overview_path_raw.c_str(), ov_w, ov_h, ov_fps)) {
        printf("RTSP+MPP OverviewRaw 初始化失败: %s\n", overview_path_raw.c_str());
        rtsp_sender_overview_raw->destroy();
        delete rtsp_sender_overview_raw;
        rtsp_sender_overview_raw = nullptr;
        use_rtsp_overview_raw = false;
      } else {
        printf("RTSP+MPP OverviewRaw 推流: %s %dx%d %.1ffps\n",
               overview_path_raw.c_str(), ov_w, ov_h, (double)ov_fps);
      }
    }
    if (use_rtsp_overview_det) {
      rtsp_sender_overview_det = new RtspMppSender();
      if (!rtsp_sender_overview_det->init(overview_path_det.c_str(), ov_w, ov_h, ov_fps)) {
        printf("RTSP+MPP OverviewDet 初始化失败: %s\n", overview_path_det.c_str());
        rtsp_sender_overview_det->destroy();
        delete rtsp_sender_overview_det;
        rtsp_sender_overview_det = nullptr;
        use_rtsp_overview_det = false;
      } else {
        printf("RTSP+MPP OverviewDet 推流: %s %dx%d %.1ffps\n",
               overview_path_det.c_str(), ov_w, ov_h, (double)ov_fps);
      }
    }
  }
#endif

  // 初始化输出视频（非 RTSP 时写文件；双路 RTSP 时不写文件）
  cv::VideoWriter writer;
  if (!output_path.empty() && !use_rtsp_mpp && !use_rtsp_dual) {
    double fps = input_fps;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer.open(output_path, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
      printf("警告: 无法打开输出视频: %s\n", output_path.c_str());
    } else {
      printf("输出视频: %s, 分辨率: %dx%d, FPS: %.1f\n",
             output_path.c_str(), width, height, fps);
    }
  }

  const int n = 6;
  printf("线程数:\t%d\n", n);
  vector<rknn_lite *> rkpool;
  dpool::ThreadPool pool(n);

  struct timeval time;
  gettimeofday(&time, nullptr);
  auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;
  long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

  if (dual_cam) {
    // ---------- 双摄：6 个 worker，前 3 路 cam0、后 3 路 cam1；每路从该路所有 slot 推流 → 约 17 FPS ----------
    cv::namedWindow("Cam0", cv::WINDOW_NORMAL);
    cv::namedWindow("Cam1", cv::WINDOW_NORMAL);
    for (int i = 0; i < n; i++) {
      rknn_lite *ptr = new rknn_lite(model_name, i % 3);
      rkpool.push_back(ptr);
    }
    std::future<int> slots[6];
    for (int i = 0; i < 3; i++) {
      cv::Mat frame;
      if (!cap.read(frame)) { printf("Cam0 无法读取初始帧\n"); return -1; }
      frame.copyTo(rkpool[i]->ori_img);
      rkpool[i]->set_tracker_stream_id(0);
      slots[i] = pool.submit(&rknn_lite::interf, rkpool[i]);
    }
    for (int i = 3; i < 6; i++) {
      cv::Mat frame;
      if (!cap1.read(frame)) { printf("Cam1 无法读取初始帧\n"); return -1; }
      frame.copyTo(rkpool[i]->ori_img);
      rkpool[i]->set_tracker_stream_id(1);
      slots[i] = pool.submit(&rknn_lite::interf, rkpool[i]);
    }
    int frames0 = 3, frames1 = 3;
#ifdef USE_RTSP_MPP
    int push_count0 = 0, push_count1 = 0;
    long last_rtsp_fps_ms = 0;
    if (use_rtsp_dual) {
      gettimeofday(&time, nullptr);
      last_rtsp_fps_ms = time.tv_sec * 1000 + time.tv_usec / 1000;
    }
#endif
    cv::Mat preview_left_raw, preview_right_raw, preview_left_det, preview_right_det;
    cv::Mat preview_stitch_raw, preview_stitch_det;
    while (true) {
      bool any = false;
      for (int i = 0; i < 6; i++) {
        if (slots[i].valid() && slots[i].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
          if (slots[i].get() != 0) goto dual_done;
          any = true;
          int cam = (i < 3) ? 0 : 1;
          cv::imshow(cam == 0 ? "Cam0" : "Cam1", rkpool[i]->ori_img);
#ifdef USE_RTSP_MPP
          // 每路从该路所有 slot 推流（非仅 slot0/slot3），6 worker 时也能到约 17 FPS
          if (cam == 0 && rtsp_sender) { rtsp_sender->push(rkpool[i]->ori_img); push_count0++; }
          if (cam == 1 && rtsp_sender1) { rtsp_sender1->push(rkpool[i]->ori_img); push_count1++; }
#endif
          cv::VideoCapture& c = (cam == 0) ? cap : cap1;
          cv::Mat frame;
          if (!c.read(frame)) { printf("视频流结束 cam%d\n", cam); goto dual_done; }
#ifdef USE_RTSP_MPP
          // raw overview 使用「原始摄像头画面」做拼接
          if (cam == 0)
            preview_left_raw = frame.clone();
          else
            preview_right_raw = frame.clone();
          // det overview 使用推理后的图像（带框）
          if (cam == 0)
            preview_left_det = rkpool[i]->ori_img.clone();
          else
            preview_right_det = rkpool[i]->ori_img.clone();
#endif
          frame.copyTo(rkpool[i]->ori_img);
          rkpool[i]->set_tracker_stream_id(cam);
          slots[i] = pool.submit(&rknn_lite::interf, rkpool[i]);
          if (cam == 0) frames0++; else frames1++;
        }
      }
      if (!any)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#ifdef USE_RTSP_MPP
      if (use_rtsp_dual && (rtsp_sender || rtsp_sender1)) {
        gettimeofday(&time, nullptr);
        long now_ms = time.tv_sec * 1000 + time.tv_usec / 1000;
        if (now_ms - last_rtsp_fps_ms >= 1000) {
          float elapsed = (now_ms - last_rtsp_fps_ms) / 1000.0f;
          printf("RTSP 推流 FPS: Cam0 %.1f  Cam1 %.1f\n", push_count0 / elapsed, push_count1 / elapsed);
          push_count0 = 0; push_count1 = 0;
          last_rtsp_fps_ms = now_ms;
        }
      }
      // 简单左右拼接原始画面，作为 OverviewRaw 预览
      if (use_rtsp_overview_raw && rtsp_sender_overview_raw &&
          !preview_left_raw.empty() && !preview_right_raw.empty()) {
        cv::Mat left_resized, right_resized;
        cv::resize(preview_left_raw, left_resized, cv::Size(640, 480));
        cv::resize(preview_right_raw, right_resized, cv::Size(640, 480));
        cv::hconcat(left_resized, right_resized, preview_stitch_raw);
        rtsp_sender_overview_raw->push(preview_stitch_raw);
      }
      // 简单左右拼接推理后画面，作为 OverviewDet 预览
      if (use_rtsp_overview_det && rtsp_sender_overview_det &&
          !preview_left_det.empty() && !preview_right_det.empty()) {
        cv::Mat left_resized, right_resized;
        cv::resize(preview_left_det, left_resized, cv::Size(640, 480));
        cv::resize(preview_right_det, right_resized, cv::Size(640, 480));
        cv::hconcat(left_resized, right_resized, preview_stitch_det);
        rtsp_sender_overview_det->push(preview_stitch_det);
      }
#endif
      if (cv::waitKey(1) == 'q') break;
    }
dual_done:
    gettimeofday(&time, nullptr);
    printf("\n平均帧率 Cam0: %.1f  Cam1: %.1f\n",
           float(frames0) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0,
           float(frames1) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);
    for (int i = 0; i < 6; i++)
      if (slots[i].valid()) slots[i].wait();
    cap1.release();
  } else {
    // ---------- 单摄：原有逻辑 ----------
    cv::namedWindow("Video Stream FPS");
    queue<std::future<int>> futs;
    int frames = 0;
    for (int i = 0; i < n; i++) {
      rknn_lite *ptr = new rknn_lite(model_name, i % 3);
      rkpool.push_back(ptr);
      cv::Mat frame;
      if (!cap.read(frame)) {
        printf("无法从视频读取初始帧\n");
        return -1;
      }
      frame.copyTo(ptr->ori_img);
      ptr->set_tracker_stream_id(2);
      futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
    }
    while (true) {
      if (futs.front().get() != 0)
        break;
      futs.pop();
      cv::Mat &show_img = rkpool[frames % n]->ori_img;
      cv::imshow("Video Stream FPS", show_img);
#ifdef USE_RTSP_MPP
      if (rtsp_sender) {
        rtsp_sender->push(show_img);
      } else
#endif
      if (writer.isOpened()) {
        writer.write(show_img);
      }
      if (cv::waitKey(1) == 'q')
        break;
      cv::Mat frame;
      if (!cap.read(frame)) {
        printf("视频流结束\n");
        break;
      }
      frame.copyTo(rkpool[frames % n]->ori_img);
      rkpool[frames % n]->set_tracker_stream_id(2);
      futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[frames++ % n])));

      // 每 2 秒打印一次平均 FPS
      gettimeofday(&time, nullptr);
      tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
      long delta = tmpTime - lopTime;
      if (delta >= 2000) {
        float elapsed_s = delta / 1000.0f;
        float fps = frames / elapsed_s;
        printf("过去 %.1f 秒平均帧率:\t%.3f 帧/s\n", elapsed_s, fps);
        lopTime = tmpTime;
        frames = 0;
      }
    }
    gettimeofday(&time, nullptr);
    printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);
    while (!futs.empty()) {
      if (futs.front().get())
        break;
      futs.pop();
    }
  }

  for (int i = 0; i < n; i++)
    delete rkpool[i];
  cap.release();
#ifdef USE_RTSP_MPP
  if (rtsp_sender) {
    rtsp_sender->destroy();
    delete rtsp_sender;
    rtsp_sender = nullptr;
  }
  if (rtsp_sender1) {
    rtsp_sender1->destroy();
    delete rtsp_sender1;
    rtsp_sender1 = nullptr;
  }
  if (rtsp_sender_overview_raw) {
    rtsp_sender_overview_raw->destroy();
    delete rtsp_sender_overview_raw;
    rtsp_sender_overview_raw = nullptr;
  }
  if (rtsp_sender_overview_det) {
    rtsp_sender_overview_det->destroy();
    delete rtsp_sender_overview_det;
    rtsp_sender_overview_det = nullptr;
  }
#endif
  if (writer.isOpened()) {
    writer.release();
  }
  cv::destroyAllWindows();
  return 0;
}
