#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_VIDEO="${1:-$ROOT_DIR/video/person.mp4}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFMPEG_ROOT="${FFMPEG_ROOT:-/usr/local/ffmpeg}"
BUILD_DIR="$ROOT_DIR/build/tools"
PROBE_SRC="$ROOT_DIR/tools/e2e_drmprime_probe.c"
PROBE_BIN="$BUILD_DIR/e2e_drmprime_probe"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "[ERR] input video not found: $INPUT_VIDEO" >&2
  exit 1
fi

if [[ ! -f "$FFMPEG_ROOT/include/libavutil/hwcontext.h" ]]; then
  echo "[ERR] FFmpeg include dir not found: $FFMPEG_ROOT/include" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "[1/4] Check local ffmpeg capability"
echo "      binary: $(command -v "$FFMPEG_BIN")"
"$FFMPEG_BIN" -version | head -n 2
if ! ("$FFMPEG_BIN" -pix_fmts 2>&1 || true) | grep -q "drm_prime"; then
  echo "[ERR] current ffmpeg does not expose drm_prime pixel format" >&2
  exit 1
fi
echo "      drm_prime pix_fmt: OK"

echo "[2/4] Verify RKMPP decode to NV12"
"$FFMPEG_BIN" -v info \
  -hwaccel rkmpp \
  -c:v h264_rkmpp \
  -i "$INPUT_VIDEO" \
  -frames:v 120 \
  -f null - |& tee "$BUILD_DIR/ffmpeg_nv12.log"

echo "[3/4] Build DRM_PRIME probe"
gcc "$PROBE_SRC" \
  -o "$PROBE_BIN" \
  -I"$FFMPEG_ROOT/include" \
  -I/usr/include/libdrm \
  -L"$FFMPEG_ROOT/lib" \
  -Wl,-rpath,"$FFMPEG_ROOT/lib" \
  -lavformat -lavcodec -lavutil -lswresample -lswscale -ldrm -lm -lz -pthread

echo "[4/4] Run DRM_PRIME probe"
stdbuf -oL -eL "$PROBE_BIN" "$INPUT_VIDEO" | tee "$BUILD_DIR/drmprime_probe.log"

if ! grep -q "receive frame format=drm_prime" "$BUILD_DIR/drmprime_probe.log"; then
  echo "[ERR] probe did not receive a drm_prime frame" >&2
  exit 1
fi

if ! grep -q "DRM layer format check: NV12 OK" "$BUILD_DIR/drmprime_probe.log"; then
  echo "[ERR] probe did not confirm DRM NV12 layout" >&2
  exit 1
fi

echo
echo "[PASS] Video DMA probe succeeded."
echo "       Step A: FFmpeg RKMPP decode path is working."
echo "       Step B: decoder can output DRM_PRIME."
echo "       Step C: DRM_PRIME layer format is NV12 and exposes a dma-buf fd."
echo
echo "Next step:"
echo "       use the printed fd path with RGA importbuffer_fd() / imcvtcolor()"
echo "       if you want to verify fd -> RGA -> RGB/BGR conversion."
