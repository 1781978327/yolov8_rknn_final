#include "http_ctrl_web_utils.h"

#include <sys/socket.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

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

std::string build_json_response(const std::string& status, const std::string& message) {
    std::ostringstream oss;
    oss << "{\"status\":\"" << status << "\"";
    if (!message.empty()) {
        oss << ",\"message\":\"" << message << "\"";
    }
    oss << ",\"timestamp\":\"" << get_timestamp() << "\"";
    oss << "}";
    return oss.str();
}

bool parse_json_body(const std::string& body, std::string& path, bool& loop) {
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
            
            <div class="btn-group">
                <button class="btn btn-start" id="btn-rtsp-start">摄像头 RTSP</button>
                <button class="btn btn-start" id="btn-rtsp-video-start">视频 RTSP</button>
                <button class="btn btn-stop" id="btn-rtsp-stop">停止推流</button>
            </div>

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
        
        document.getElementById('btn-rtsp-start').onclick = () => fetch('/api/rtsp/start', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'success'));
        document.getElementById('btn-rtsp-video-start').onclick = () => fetch('/api/rtsp/video/start', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'success'));
        document.getElementById('btn-rtsp-stop').onclick = () => fetch('/api/rtsp/stop', {method: 'POST'}).then(r => r.json()).then(d => addLog(d.message, 'info'));

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

void send_response(int client_fd, const std::string& content, const std::string& content_type, int status_code) {
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
