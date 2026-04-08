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

## 当前视频链路

- `main_http_ctrl` 的视频文件主链路已切为：
  - `FFmpeg h264_rkmpp/hevc_rkmpp` 解码
  - 优先输出 `AV_PIX_FMT_DRM_PRIME`
  - 从 `AVDRMFrameDescriptor` 提取 `dma-buf fd`
  - `rkpool[i]->set_video_dmabuf_frame(...)`
  - `RGA -> RKNN input_mem`
  - 推理 / 绘制后的 `BGR` 帧再回到 `dma-buf` staging
  - `RGA -> NV12 -> MPP encoder -> RTSP`
- 如果 `DRM_PRIME` 不可用，代码仍保留 `NV12` 与 `OpenCV/FFmpeg` 回退路径
- `cam3` 现在支持两种工作方式：
  - 仅视频裸流：`/api/video/start` + `/api/rtsp/video/start`
  - 视频推理流：在上面基础上再调用 `/api/inference/on`

## 当前摄像头链路

- 摄像头采集优先使用 `V4L2 DMABUF`
- 摄像头推理前处理主链路已切为：
  - `camera dma-buf fd`
  - `RGA -> RKNN input_mem`
- 摄像头显示 / 叠框仍保留 `cv::Mat(BGR)` 路径
- 摄像头 RTSP 推流主链路已切为：
  - 推理 / 绘制后的 `BGR` 帧写回 `dma-buf` staging
  - `RGA -> NV12 -> MPP encoder -> RTSP`
- 若摄像头 `DMABUF` 不可用，代码仍会回退到 `OpenCV V4L2`

最近一次本机回归结果：

- `cam0` / `cam1`：`h264 640x480 @ 30fps`
- `cam3`：`h264 1920x1080 @ 25fps`
- 本机 `ffprobe` / `ffmpeg` 均可直接拉流
- `main_http_ctrl` 视频推理日志可见 `frame_fmt=drm_prime(179) drm_valid=1`
- 视频模式推理前处理日志可见 `video drm_fd -> RGA -> input_mem`
- 摄像头模式推理前处理日志可见 `camera dmabuf -> RGA -> input_mem`
- 推理后回 DMA 再推流日志可见 `RTSP-DMA] BGR frame -> dma-bgr -> RGA -> NV12 -> MPP encoder`

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

关于模型路径的建议：

- 程序默认模型路径是相对路径 `../model/RK3588/yolov8s.rknn`
- 相对路径是按“服务进程启动时的工作目录”解析，不一定等于项目根目录
- 如果你看到 `未找到可用模型(.rknn)`，优先在 `/api/inference/on` 中使用绝对路径 `model=...`

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

### 本地联调最短链路（摄像头 + 推理跟踪 + 抓图 + 推流）

下面这组命令是本机实测可通的最短链路，建议直接复制执行：

```bash
BASE=http://127.0.0.1:8091
MODEL=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn
LABELS=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/coco_80_labels_list.txt

# 1) 清理状态并切回摄像头模式
curl -X POST "$BASE/api/rtsp/stop"
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST "$BASE/api/video/stop"

# 2) 选摄像头（0 或 1）
curl -X POST "$BASE/api/camera/0"

# 3) 开推理+跟踪（关键：模型/标签建议用绝对路径）
curl -X POST -G "$BASE/api/inference/on" \
  --data-urlencode "track=1" \
  --data-urlencode "tracker=bytetrack" \
  --data-urlencode "model=$MODEL" \
  --data-urlencode "labels=$LABELS"

# 4) 等首帧并抓图
sleep 1
curl "$BASE/api/frame?track=1" -o cam0.jpg
file cam0.jpg

# 5) 开 RTSP 推流
curl -X POST "$BASE/api/rtsp/start"
curl "$BASE/api/status"
```

如果 `cam0.jpg` 不是 JPEG（例如只有几十字节），通常是返回了 JSON 错误信息，可以这样检查：

```bash
file cam0.jpg
cat cam0.jpg
```

常见情况是 `{"status":"error","message":"暂无帧数据",...}`，此时建议：

- 先看 `curl "$BASE/api/status"` 中 `model_loaded` 是否为 `true`
- 再切到另一只摄像头测试：`curl -X POST "$BASE/api/camera/1"` 后重试抓图

### 查看状态

```bash
curl $BASE/api/status
curl $BASE/api/video/status
curl $BASE/api/tracker
curl $BASE/api/threshold/get
curl "$BASE/api/detection/count?cam=0"
curl "$BASE/api/detection/count?cam=1"
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

切换当前摄像头编号（仅切换状态字段，不会修改固定 RTSP 路由）：

```bash
curl -X POST $BASE/api/camera/0
curl -X POST $BASE/api/camera/1
```

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

如果你只想推原视频裸流，不开推理，直接启动视频 RTSP：

```bash
curl -X POST $BASE/api/rtsp/video/start
```

如果要做视频推理，再启动推理：

```bash
curl -X POST "$BASE/api/inference/on?track=0&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn"
```

启动视频 RTSP：

```bash
curl -X POST $BASE/api/rtsp/video/start
```

视频 RTSP 地址：

- `rtsp://127.0.0.1:8554/cam3`

当前实现说明：

- 视频文件模式下，worker 输入优先走 `DRM_PRIME dma-buf`
- HTTP 预览和叠框显示仍保留 `cv::Mat(BGR)` 路径
- RTSP 推流编码前会优先把推理后的 `BGR` 帧写回 `dma-buf` staging，再由 `RGA + MPP` 推出
- 因此同一帧会同时存在：
  - 一条用于推理的 `drm fd`
  - 一条用于显示/叠框的 `BGR Mat`

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

- `/api/frame` 只有在视频流或摄像头流已经启动并产生过当前帧时才会返回 JPEG
- 在空闲态调用 `/api/frame` 会返回 `503` 与 `暂无帧数据`
- 建议抓图后用 `file /tmp/frame.jpg` 或 `curl -i` 检查 `Content-Type`，避免把错误 JSON 当成 JPG
- `redraw=1` 是调试开关
- 推理线程本身已经在 `ori_img` 上画过框
- 使用 `redraw=1` 可能看到文字或框叠加

## 其他常用接口

```bash
curl -X POST "$BASE/api/threshold/set?value=0.55"
curl -X POST "$BASE/api/threshold/set?value=0.40"
curl -X POST "$BASE/api/threshold/set?value=0.70"
curl -X POST "$BASE/api/threshold/set?value=0.50"
curl "$BASE/api/detection/count?cam=0"
curl "$BASE/api/detection/count?cam=1"
curl -X POST $BASE/api/tracker/1
curl -X POST $BASE/api/tracker/0
curl -X POST $BASE/api/camera/0
curl -X POST $BASE/api/camera/1
curl -X POST $BASE/api/camera/2
curl -X POST $BASE/api/rtsp/stop
curl -X POST "$BASE/api/inference/off"
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST $BASE/api/video/stop
curl $BASE/
```

说明：

- `/api/status` 会返回 `model_loaded`、`video_mode`、`rtsp_streaming`、`tracker_backend`、`tracker_skip_frames` 等完整状态
- `/api/frame` 在空闲态会返回 `503` 与 `暂无帧数据`
- `/api/threshold/set` 建议配合 `/api/threshold/get` 立即确认是否生效
- `/api/camera/2` 当前会返回成功，但摄像头 RTSP 仍固定使用 `cam0=/dev/video0`、`cam1=/dev/video2`

## 用 ffplay / ffprobe 验证推流

视频模式：

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json rtsp://127.0.0.1:8554/cam3

ffmpeg -v error -rtsp_transport tcp \
  -i rtsp://127.0.0.1:8554/cam3 -t 3 -f null -

ffplay  -rtsp_transport tcp -fflags nobuffer -flags low_delay rtsp://127.0.0.1:8554/cam3
```

双摄模式：

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json rtsp://127.0.0.1:8554/cam0

ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json rtsp://127.0.0.1:8554/cam1

ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/cam0
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1
```

说明：

- `cam3` 启动后如果立刻探测，首次偶发 `404` 属于发布路径尚未稳定，通常等待约 `1` 秒再探测即可
- 视频 RTSP 当前仍可能出现 `non monotonically increasing dts` 警告，但不影响本机 `ffprobe` / `ffmpeg` 拉流与解码

## API 回归建议

建议按下面顺序做一次完整回归：

```bash
BASE=http://127.0.0.1:8091

# 1. 只读接口
curl $BASE/api/status
curl $BASE/api/video/status
curl $BASE/api/tracker
curl $BASE/api/threshold/get
curl "$BASE/api/detection/count?cam=0"
curl "$BASE/api/detection/count?cam=1"

# 2. 阈值接口
curl -X POST "$BASE/api/threshold/set?value=0.4"
curl $BASE/api/threshold/get
curl -X POST "$BASE/api/threshold/set?value=0.5"
curl $BASE/api/threshold/get

# 3. 视频文件 + YOLO
curl -X POST $BASE/api/rtsp/stop
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST $BASE/api/video/stop
curl -X POST $BASE/api/video/start \
  -H 'Content-Type: application/json' \
  -d '{"path":"/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/person.mp4","loop":true}'
curl -X POST "$BASE/api/inference/on?track=0&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn"
curl -X POST $BASE/api/rtsp/video/start
curl $BASE/api/status
curl $BASE/api/video/status

# 4. 摄像头 + 跟踪
curl -X POST $BASE/api/rtsp/stop
curl -X POST "$BASE/api/inference/off?unload=1"
curl -X POST $BASE/api/video/stop
curl -X POST "$BASE/api/inference/on?track=1&tracker=bytetrack&model=/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/model/RK3588/yolov8s.rknn"
curl -X POST $BASE/api/rtsp/start
curl "$BASE/api/frame?track=1&redraw=1" --output /tmp/frame_track.jpg
```

## 用工具验证 DRM_PRIME

最小探针：

```bash
./tools/test_video_dma.sh
```

它会验证：

- `FFmpeg RKMPP` 能否把本地 MP4 解到 `NV12`
- 解码器能否输出 `AV_PIX_FMT_DRM_PRIME`
- `DRM_PRIME` 图层格式是否为 `NV12`
- 是否能拿到有效 `dma-buf fd`

独立 RTSP 推流测试：

```bash
cmake --build build -j4 --target drmprime_rtsp_push_test

./build/drmprime_rtsp_push_test \
  /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/person.mp4 \
  rtsp://127.0.0.1:8554/drmprime-test \
  --loop
```

本机验证：

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of json rtsp://127.0.0.1:8554/drmprime-test
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
