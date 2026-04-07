#ifndef FFMPEG_RKMPP_READER_H
#define FFMPEG_RKMPP_READER_H

#include <string>

#include <opencv2/core/mat.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
}

class FFmpegRkmppReader
{
public:
    struct FrameInfo {
        bool valid = false;
        int fd = -1;
        bool nv12_valid = false;
        cv::Mat nv12_packed;
        int size = 0;
        int width = 0;
        int height = 0;
        int wstride = 0;
        int hstride = 0;
        uint32_t drm_format = 0;
    };

    FFmpegRkmppReader();
    ~FFmpegRkmppReader();

    bool Open(const std::string& file_path);
    bool ReadFrame(cv::Mat& bgr_frame);
    bool ReadFrame(cv::Mat& bgr_frame, FrameInfo* frame_info);
    void Close();

    bool IsOpen() const { return format_ctx_ != nullptr && codec_ctx_ != nullptr; }
    int Width() const { return width_; }
    int Height() const { return height_; }
    double Fps() const { return fps_; }
    const std::string& DecoderName() const { return decoder_name_; }

private:
    bool ReceiveFrame(cv::Mat& bgr_frame, FrameInfo* frame_info);
    bool ExtractDrmPrimeInfo(const AVFrame* frame, FrameInfo* frame_info);
    bool ConvertFrameToBgr(const AVFrame* frame, cv::Mat& bgr_frame);
    bool ConvertDrmPrimeFrameToBgr(const AVFrame* frame, cv::Mat& bgr_frame);
    static enum AVPixelFormat PreferDrmPrime(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);
    static const char* SelectDecoderName(AVCodecID codec_id);

private:
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    const AVCodec* codec_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    int video_stream_index_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    bool sent_eof_ = false;
    std::string decoder_name_;
};

#endif  // FFMPEG_RKMPP_READER_H
