#ifndef FFMPEG_WITH_MPP
#define FFMPEG_WITH_MPP

#include <string>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/time.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext_drm.h>
#include <libdrm/drm_fourcc.h>
}

extern char errInfo[200];
void print_error(int line, int res, const std::string& selfInfo);

#endif
