#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT}/build/rknn_yolov8_demo"

if [[ ! -x "${BIN}" ]]; then
  echo "未找到可执行文件: ${BIN}"
  echo "请先在 ${ROOT}/build 下编译: cmake .. && make"
  exit 1
fi

# 让运行时优先使用本工程 build/lib 里的 FFmpeg so，避免 libswresample.so.4 找不到
export LD_LIBRARY_PATH="${ROOT}/build/lib:${LD_LIBRARY_PATH:-}"

exec "${BIN}" "$@"

