# YOLOv8 RK3588 HTTP 控制服务

基于 RK3588 NPU 的双路视觉服务，支持：
- 摄像头推理开关
- RTSP 裸流/推理流
- 目标跟踪开关
- 单帧抓图接口
- 视频文件模式

默认端口：
- HTTP: `8091`
- RTSP: `8554`（由 `mediamtx` 提供）

## 1. 最近修复的 Bug（2026-04-05）

以下问题已在代码中修复并回归验证：

1. `RTSP` 发包内存泄漏  
问题：每帧发包分配后未正确释放，长跑后内存增长，可能触发 OOM。  
修复：改为 `av_new_packet` + `av_packet_unref`，并完善失败路径释放。

2. `/api/rtsp/stop` 只改标志位，不销毁 sender  
问题：停止后可能仍能连到“空壳流”，表现为状态不一致。  
修复：`stop` 时在锁内立即销毁 sender，停止后端点应返回 404。

3. RTSP 重启流程中的上下文释放问题（双重释放风险）  
问题：`hwdevice/hwframe` 引用管理不正确，重启时有概率触发崩溃。  
修复：改为 `av_buffer_ref` 管理 codec 上下文引用，统一在销毁路径释放。

4. MPP 资源未完全回收  
问题：编码配置和上下文回收不完整。  
修复：补齐 `mpp_enc_cfg_deinit` 与 `mpp_destroy`。

5. 增加排障日志  
新增 `start/stop` 关键日志，打印 sender 指针状态和进程 RSS，便于定位异常。

### 1.1 RTSP 卡顿优化记录（2026-04-05）

现象（优化前）：
- 服务端日志显示 `RTSP: 30` 左右，但 VLC 端主观观感明显卡顿、偶发“糊”和不顺滑。

已实施优化：
- 修复 RTSP 发送包时间戳字段在 `av_new_packet()` 后被覆盖的问题（`pts/dts/duration` 在分配后重新写入）。
- 将摄像头 RTSP 输出目标帧率从 `25` 调整为 `30`，与采集端速率保持一致，减少时基不匹配导致的播放抖动。

验证结果：
- `ffprobe` 流信息：`avg_frame_rate=30/1`。
- 12 秒包计数实测：`cam0/cam1` 约 `29~30 fps`。
- 本地 VLC 主观观感较优化前明显改善（“没那么卡”）。

## 2. 编译与启动

### 2.1 启动 RTSP 服务器（终端 A）

```bash
cd /home/orangepi/Desktop/rkmpp编码案例
./mediamtx
```

### 2.2 编译并运行服务（终端 B）

```bash
cd /home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/build
cmake .. && make -j$(nproc)
./rknn_http_ctrl --port 8091
```

### 2.3 可选参数

```bash
./rknn_http_ctrl --port 8091 --rtsp-host 127.0.0.1 --rtsp-port 8554
```

或环境变量：

```bash
export RTSP_HOST=127.0.0.1
export RTSP_PORT=8554
./rknn_http_ctrl --port 8091
```

## 3. API 总览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 获取全量运行状态 |
| POST | `/api/threshold/set?value=0.6` | 设置检测阈值（0~1） |
| GET | `/api/threshold/get` | 获取检测阈值 |
| GET | `/api/detection/count?cam=0` | 获取指定摄像头检测数量 |
| GET | `/api/frame?track=1` | 获取一帧 JPEG 图像 |
| GET | `/api/tracker` | 获取跟踪开关状态 |
| POST | `/api/tracker/1` | 开启跟踪（`/0` 关闭） |
| POST | `/api/inference/on?...` | 开启推理（可带 model/labels） |
| POST | `/api/inference/off?...` | 关闭推理（可卸载模型） |
| POST | `/api/camera/{id}` | 切换当前摄像头（0~2） |
| POST | `/api/rtsp/start` | 开启 RTSP（摄像头模式） |
| POST | `/api/rtsp/video/start` | 开启 RTSP（视频模式） |
| POST | `/api/rtsp/stop` | 停止 RTSP |
| GET | `/api/video/status` | 视频模式状态 |
| POST | `/api/video/start` | 启动视频文件模式 |
| POST | `/api/video/stop` | 停止视频文件模式 |
| GET | `/` or `/index.html` | Web 控制页 |

## 4. API 详细调用方法（含示例）

以下示例默认服务地址为：

```bash
HOST=http://127.0.0.1:8091
```

### 4.1 状态接口

```bash
curl -sS $HOST/api/status
```

返回示例（字段会随运行状态变化）：

```json
{
  "running": true,
  "inference_enabled": false,
  "model_loaded": false,
  "model_loading": false,
  "model_switching": false,
  "active_jobs": 0,
  "model_path": "../model/RK3588/yolov8s.rknn",
  "label_path": ".../model/coco_80_labels_list.txt",
  "rtsp_url_cam0": "rtsp://127.0.0.1:8554/cam0",
  "rtsp_url_cam1": "rtsp://127.0.0.1:8554/cam1",
  "rtsp_url_video": "rtsp://127.0.0.1:8554/cam3",
  "tracker_enabled": false,
  "detection_count_cam0": 0,
  "detection_count_cam1": 0,
  "threshold": 0.5,
  "video_mode": false,
  "rtsp_streaming": false
}
```

### 4.2 检测阈值

```bash
curl -sS -X POST "$HOST/api/threshold/set?value=0.6"
curl -sS $HOST/api/threshold/get
```

### 4.3 检测数量

```bash
curl -sS "$HOST/api/detection/count?cam=0"
curl -sS "$HOST/api/detection/count?cam=1"
```

### 4.4 跟踪开关

查询：

```bash
curl -sS $HOST/api/tracker
```

开启/关闭：

```bash
curl -sS -X POST $HOST/api/tracker/1
curl -sS -X POST $HOST/api/tracker/0
```

也支持：

```bash
curl -sS -X POST $HOST/api/tracker/on
```

### 4.5 推理开关

开启推理（默认开启跟踪，除非明确 `track=0`）：

```bash
curl -sS -X POST "$HOST/api/inference/on"
curl -sS -X POST "$HOST/api/inference/on?track=1"
curl -sS -X POST "$HOST/api/inference/on?track=0"
```

指定模型/标签：

```bash
curl -sS -X POST "$HOST/api/inference/on?model=../model/RK3588/yolov8n.rknn"
curl -sS -X POST "$HOST/api/inference/on?labels=/tmp/my_labels.txt"
curl -sS -X POST "$HOST/api/inference/on?model=../model/RK3588/yolov8s.rknn&labels=/tmp/my_labels.txt&track=1"
```

关闭推理：

```bash
curl -sS -X POST "$HOST/api/inference/off"
```

关闭并卸载模型：

```bash
curl -sS -X POST "$HOST/api/inference/off?unload=1"
```

关闭并设置“下次加载”的模型/标签：

```bash
curl -sS -X POST "$HOST/api/inference/off?model=../model/RK3588/yolov8m.rknn"
curl -sS -X POST "$HOST/api/inference/off?labels=/tmp/my_labels.txt"
```

恢复默认标签（下次加载）：

```bash
curl -sS -X POST "$HOST/api/inference/off?labels=default"
```

注意：
- 推理开启时如果要切换模型/标签，接口会返回 409，需先调用 `inference/off`。

### 4.6 摄像头切换

```bash
curl -sS -X POST $HOST/api/camera/0
curl -sS -X POST $HOST/api/camera/1
curl -sS -X POST $HOST/api/camera/2
```

### 4.7 RTSP 控制

开启（摄像头双路）：

```bash
curl -sS -X POST $HOST/api/rtsp/start
```

开启（视频模式专用）：

```bash
curl -sS -X POST $HOST/api/rtsp/video/start
```

停止：

```bash
curl -sS -X POST $HOST/api/rtsp/stop
```

拉流地址：
- `rtsp://127.0.0.1:8554/cam0`
- `rtsp://127.0.0.1:8554/cam1`
- `rtsp://127.0.0.1:8554/cam3`（视频模式）

### 4.8 视频文件模式

状态：

```bash
curl -sS $HOST/api/video/status
```

启动：

```bash
curl -sS -X POST $HOST/api/video/start \
  -H "Content-Type: application/json" \
  -d '{"path":"/home/orangepi/Desktop/web/yolov8-rk3588-cpp-3-15/video/demo.mp4","loop":true}'
```

停止：

```bash
curl -sS -X POST $HOST/api/video/stop
```

### 4.9 抓取单帧图片

抓取一帧（JPEG）：

```bash
curl -sS "$HOST/api/frame?track=1" -o /tmp/frame.jpg
file /tmp/frame.jpg
```

参数说明：
- `track=1`：允许跟踪框输出
- `track=0`：不需要跟踪框
- `redraw=1` 或 `overlay=1`：强制二次叠加绘制（调试用）

说明：
- 常规情况下不建议 `redraw=1`，否则可能出现“重复框”。
- 若暂无帧数据，接口返回 `503` 和 JSON 错误信息。

### 4.10 Web 页面

```bash
curl -sS $HOST/
curl -sS $HOST/index.html
```

浏览器访问：

```text
http://<设备IP>:8091/
```

## 5. 推荐联调流程

```bash
HOST=http://127.0.0.1:8091

# 1) 裸流
curl -sS -X POST "$HOST/api/inference/off?unload=1"
curl -sS -X POST "$HOST/api/rtsp/start"
curl -sS "$HOST/api/status"

# 2) 开推理
curl -sS -X POST "$HOST/api/inference/on?track=1"
sleep 2
curl -sS "$HOST/api/status"

# 3) 抓图
curl -sS "$HOST/api/frame?track=1" -o /tmp/frame_infer.jpg
file /tmp/frame_infer.jpg

# 4) 停推理保留推流
curl -sS -X POST "$HOST/api/inference/off?unload=1"
curl -sS "$HOST/api/status"

# 5) 停推流
curl -sS -X POST "$HOST/api/rtsp/stop"
curl -sS "$HOST/api/status"
```

## 6. 常见问题

1. `api/frame` 返回 503  
原因：当前还没有最新帧。  
处理：先 `rtsp/start` 或 `inference/on`，等待 1~2 秒再抓图。

2. RTSP 停止后仍能播放  
正常预期是停止后 404；若不是，优先确认是否连到了旧地址或外部缓存。

3. 切换模型返回 409  
先执行 `POST /api/inference/off`，再带新模型参数调用 `POST /api/inference/on`。

4. 连不上 8091  
确认 `rknn_http_ctrl` 在前台运行且端口未冲突。

## 7. 代码位置

主要接口实现：
- `src/main_http_ctrl.cc`

RTSP 发送实现：
- `src/rtsp_mpp_sender.cpp`
