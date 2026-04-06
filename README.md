# YOLOv8 RK3588 HTTP/RTSP Service

基于 RK3588 NPU 的双路视觉推理服务，提供：

- 双摄像头推理
- 视频文件推理
- RTSP 推流
- HTTP 控制接口
- 单帧 JPEG 抓图
- 可切换跟踪算法：`ByteTrack` / `DeepSORT`

默认端口：

- HTTP: `8091`
- RTSP: `8554`，需外部 RTSP 服务器接收，例如 `mediamtx`

## 目录说明

```text
yolov8-rk3588-cpp-3-15/
├── src/                 # 主程序与 HTTP 控制服务
├── include/             # RKNN 推理与公共头文件
├── bytetrack/           # ByteTrack
├── deepsort/            # DeepSORT + ReID
├── model/RK3588/        # YOLO / ReID 模型
└── video/               # 测试视频
```

常用模型：

- 检测模型：`model/RK3588/yolov8s.rknn`
- ReID 模型：`model/RK3588/osnet_x0_25_market.rknn`

## 依赖

- RK3588 / OrangePi 5B
- JDK 与前端不是本项目运行必需
- OpenCV 4
- Eigen3
- RGA
- FFmpeg 60 + RKMPP
- RTSP 服务器，例如 `mediamtx`

## 编译

推荐使用 `Release` 构建：

```bash
cd /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15
mkdir -p build_release
cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2 rknn_http_ctrl rknn_yolov8_demo
```

开发调试也可以继续用原来的 `build/`：

```bash
cd /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/build
cmake ..
make -j2 rknn_http_ctrl rknn_yolov8_demo
```

## 启动 RTSP 服务器

服务端只负责“推”，不负责创建 RTSP 服务端，所以需要先启动 `mediamtx` 或其他 RTSP 服务器。

例如：

```bash
./mediamtx
```

确认 `8554` 已监听：

```bash
ss -ltn | grep 8554
```

## 启动 HTTP 服务

默认启动：

```bash
cd /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/build_release
sudo ./rknn_http_ctrl
```

指定默认跟踪器与 ReID：

```bash
sudo ./rknn_http_ctrl \
  --tracker-backend deepsort \
  --reid-model /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn
```

也支持环境变量：

```bash
sudo TRACKER_BACKEND=deepsort \
TRACKER_REID_MODEL=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn \
./rknn_http_ctrl
```

摄像头目前默认打开：

- `cam0 -> /dev/video0`
- `cam1 -> /dev/video2`

如需确认设备号：

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

## HTTP API 快速使用

先设一个变量：

```bash
BASE=http://127.0.0.1:8091
```

### 查看状态

```bash
curl $BASE/api/status
curl $BASE/api/video/status
curl $BASE/api/tracker
curl $BASE/api/threshold/get
```

### 开启摄像头模式

ByteTrack：

```bash
curl -X POST "$BASE/api/inference/on?track=1&tracker=bytetrack&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn"
curl -X POST $BASE/api/rtsp/start
```

DeepSORT：

```bash
curl -X POST "$BASE/api/inference/on?track=1&tracker=deepsort&tracker_skip=2&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn&reid_model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn"
curl -X POST $BASE/api/rtsp/start
```

RTSP 地址：

- `rtsp://127.0.0.1:8554/cam0`
- `rtsp://127.0.0.1:8554/cam1`

### 开启视频文件模式

先停掉旧状态：

```bash
curl -X POST $BASE/api/rtsp/stop
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST $BASE/api/video/stop
```

启动视频：

```bash
curl -X POST $BASE/api/video/start \
  -H 'Content-Type: application/json' \
  -d '{"path":"/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/person.mp4","loop":true}'
```

启动推理：

```bash
curl -X POST "$BASE/api/inference/on?track=1&tracker=deepsort&tracker_skip=2&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn&reid_model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn"
```

启动视频 RTSP：

```bash
curl -X POST $BASE/api/rtsp/video/start
```

视频 RTSP 地址：

- `rtsp://127.0.0.1:8554/cam3`

## 跟踪器说明

### ByteTrack

- 纯检测框 + IoU 跟踪
- 性能高
- 目前已改为按流共享实例，视频模式和双摄模式下 ID 会比早期版本稳定

### DeepSORT

- 额外使用 ReID 模型
- 支持 HTTP 参数 `tracker_skip`
- 已改为按流共享实例
- 跳帧时会用 IoU 复用上一轮带 ID 的框

### `tracker_skip` 说明

仅对 `DeepSORT` 生效：

- `0`：每帧做 ReID
- `1`：隔 1 帧
- `2`：隔 2 帧

例如：

```bash
curl -X POST "$BASE/api/inference/on?track=1&tracker=deepsort&tracker_skip=2&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn&reid_model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn"
```

当前值可在状态接口中查看：

```bash
curl $BASE/api/status
```

返回字段：

- `tracker_backend`
- `tracker_reid_model`
- `tracker_skip_frames`

## 抓图接口

抓取当前一帧：

```bash
curl "$BASE/api/frame" --output /tmp/frame.jpg
```

带调试重绘：

```bash
curl "$BASE/api/frame?track=1&redraw=1" --output /tmp/frame_track.jpg
```

注意：

- `redraw=1` 是调试开关
- 推理线程本身已经在 `ori_img` 上画过框
- 使用 `redraw=1` 可能看到文字或框叠加

## 其他常用接口

```bash
curl -X POST "$BASE/api/threshold/set?value=0.55"
curl "$BASE/api/detection/count?cam=0"
curl "$BASE/api/detection/count?cam=1"
curl -X POST $BASE/api/tracker/1
curl -X POST $BASE/api/tracker/0
curl -X POST $BASE/api/camera/0
curl -X POST $BASE/api/camera/1
curl -X POST $BASE/api/rtsp/stop
curl -X POST "$BASE/api/inference/off"
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST $BASE/api/video/stop
```

## 用 ffplay / ffprobe 验证推流

视频模式：

```bash
ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/cam3
ffplay  -rtsp_transport tcp -fflags nobuffer -flags low_delay rtsp://127.0.0.1:8554/cam3
```

双摄模式：

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/cam0
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1
```

## 命令行程序 `rknn_yolov8_demo`

默认 ByteTrack：

```bash
cd /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/build_release
./rknn_yolov8_demo /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn
```

DeepSORT：

```bash
TRACKER_BACKEND=deepsort \
TRACKER_REID_MODEL=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/osnet_x0_25_market.rknn \
./rknn_yolov8_demo /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn
```

## 已知说明

- DeepSORT 相比 ByteTrack 更吃性能，`1080p + RTSP + ReID` 下建议优先尝试 `tracker_skip=1/2`
- `DeepSORT` 与 `ByteTrack` 当前都已按流共享实例，不再是每个 worker 各自维护一套状态
- 如果切换模型、标签或跟踪算法，建议先：

```bash
curl -X POST "$BASE/api/inference/off?unload=1"
```

再重新 `inference/on`
