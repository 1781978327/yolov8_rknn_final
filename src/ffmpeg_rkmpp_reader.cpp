#include "ffmpeg_rkmpp_reader.h"

#include <cstring>

#include <opencv2/imgproc.hpp>

namespace {

static cv::Mat MakeNv12Mat(const AVFrame* frame)
{
    cv::Mat nv12(frame->height + frame->height / 2, frame->width, CV_8UC1);

    for (int y = 0; y < frame->height; ++y) {
        memcpy(nv12.ptr(y), frame->data[0] + y * frame->linesize[0], frame->width);
    }

    for (int y = 0; y < frame->height / 2; ++y) {
        memcpy(nv12.ptr(frame->height + y), frame->data[1] + y * frame->linesize[1], frame->width);
    }

    return nv12;
}

static cv::Mat MakeI420Mat(const AVFrame* frame)
{
    cv::Mat i420(frame->height + frame->height / 2, frame->width, CV_8UC1);

    uint8_t* dst_y = i420.ptr<uint8_t>(0);
    uint8_t* dst_u = dst_y + frame->width * frame->height;
    uint8_t* dst_v = dst_u + (frame->width / 2) * (frame->height / 2);

    for (int y = 0; y < frame->height; ++y) {
        memcpy(dst_y + y * frame->width, frame->data[0] + y * frame->linesize[0], frame->width);
    }
    for (int y = 0; y < frame->height / 2; ++y) {
        memcpy(dst_u + y * (frame->width / 2), frame->data[1] + y * frame->linesize[1], frame->width / 2);
        memcpy(dst_v + y * (frame->width / 2), frame->data[2] + y * frame->linesize[2], frame->width / 2);
    }

    return i420;
}

}  // namespace

FFmpegRkmppReader::FFmpegRkmppReader() {}

FFmpegRkmppReader::~FFmpegRkmppReader()
{
    Close();
}

bool FFmpegRkmppReader::Open(const std::string& file_path)
{
    Close();

    format_ctx_ = avformat_alloc_context();
    if (!format_ctx_) {
        return false;
    }

    if (avformat_open_input(&format_ctx_, file_path.c_str(), nullptr, nullptr) != 0) {
        Close();
        return false;
    }
    if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
        Close();
        return false;
    }

    video_stream_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index_ < 0) {
        Close();
        return false;
    }

    AVStream* stream = format_ctx_->streams[video_stream_index_];
    decoder_name_ = SelectDecoderName(stream->codecpar->codec_id);
    if (decoder_name_.empty()) {
        Close();
        return false;
    }

    codec_ = avcodec_find_decoder_by_name(decoder_name_.c_str());
    if (!codec_) {
        Close();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec_);
    if (!codec_ctx_) {
        Close();
        return false;
    }
    if (avcodec_parameters_to_context(codec_ctx_, stream->codecpar) < 0) {
        Close();
        return false;
    }

    AVDictionary* opts = nullptr;
    av_dict_set_int(&opts, "buf_mode", 1, 0);
    if (avcodec_open2(codec_ctx_, codec_, &opts) < 0) {
        av_dict_free(&opts);
        Close();
        return false;
    }
    av_dict_free(&opts);

    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !packet_) {
        Close();
        return false;
    }

    width_ = codec_ctx_->width;
    height_ = codec_ctx_->height;
    fps_ = av_q2d(stream->avg_frame_rate);
    if (fps_ <= 0.0) {
        fps_ = av_q2d(stream->r_frame_rate);
    }
    if (fps_ <= 0.0) {
        fps_ = 25.0;
    }
    sent_eof_ = false;
    return true;
}

bool FFmpegRkmppReader::ReadFrame(cv::Mat& bgr_frame)
{
    return ReadFrame(bgr_frame, nullptr);
}

bool FFmpegRkmppReader::ReadFrame(cv::Mat& bgr_frame, FrameInfo* frame_info)
{
    if (!format_ctx_ || !codec_ctx_ || !frame_ || !packet_) {
        return false;
    }

    while (true) {
        if (ReceiveFrame(bgr_frame, frame_info)) {
            return true;
        }

        if (sent_eof_) {
            return false;
        }

        int ret = av_read_frame(format_ctx_, packet_);
        if (ret < 0) {
            avcodec_send_packet(codec_ctx_, nullptr);
            sent_eof_ = true;
            continue;
        }

        if (packet_->stream_index != video_stream_index_) {
            av_packet_unref(packet_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, packet_);
        av_packet_unref(packet_);
        if (ret < 0) {
            return false;
        }
    }
}

void FFmpegRkmppReader::Close()
{
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (packet_) {
        av_packet_free(&packet_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
    }

    codec_ = nullptr;
    video_stream_index_ = -1;
    width_ = 0;
    height_ = 0;
    fps_ = 0.0;
    sent_eof_ = false;
    decoder_name_.clear();
}

bool FFmpegRkmppReader::ReceiveFrame(cv::Mat& bgr_frame, FrameInfo* frame_info)
{
    static int decode_log_counter = 0;
    if (frame_info) {
        *frame_info = FrameInfo{};
    }

    int ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    }
    if (ret < 0) {
        return false;
    }

    if (frame_info && frame_->format == AV_PIX_FMT_NV12) {
        frame_info->nv12_valid = true;
        frame_info->size = frame_->width * frame_->height * 3 / 2;
        frame_info->width = frame_->width;
        frame_info->height = frame_->height;
        frame_info->wstride = frame_->width;
        frame_info->hstride = frame_->height;
        frame_info->nv12_packed = MakeNv12Mat(frame_);
    }

    if ((++decode_log_counter % 120) == 0) {
        printf("[VideoDecode] frame_fmt=%d nv12_valid=%d decoder=%s\n",
               frame_->format,
               frame_info && frame_info->nv12_valid ? 1 : 0,
               decoder_name_.c_str());
    }

    const bool ok = ConvertFrameToBgr(frame_, bgr_frame);
    av_frame_unref(frame_);
    return ok;
}

bool FFmpegRkmppReader::ConvertFrameToBgr(const AVFrame* frame, cv::Mat& bgr_frame)
{
    if (!frame || frame->width <= 0 || frame->height <= 0) {
        return false;
    }

    if (frame->format == AV_PIX_FMT_NV12) {
        cv::Mat nv12 = MakeNv12Mat(frame);
        cv::cvtColor(nv12, bgr_frame, cv::COLOR_YUV2BGR_NV12);
        return true;
    }
    if (frame->format == AV_PIX_FMT_YUV420P) {
        cv::Mat i420 = MakeI420Mat(frame);
        cv::cvtColor(i420, bgr_frame, cv::COLOR_YUV2BGR_I420);
        return true;
    }

    return false;
}

const char* FFmpegRkmppReader::SelectDecoderName(AVCodecID codec_id)
{
    switch (codec_id) {
    case AV_CODEC_ID_H264:
        return "h264_rkmpp";
    case AV_CODEC_ID_HEVC:
        return "hevc_rkmpp";
    case AV_CODEC_ID_MJPEG:
        return "mjpeg_rkmpp";
    case AV_CODEC_ID_VP8:
        return "vp8_rkmpp";
    case AV_CODEC_ID_VP9:
        return "vp9_rkmpp";
    case AV_CODEC_ID_MPEG2VIDEO:
        return "mpeg2_rkmpp";
    case AV_CODEC_ID_MPEG4:
        return "mpeg4_rkmpp";
    default:
        return nullptr;
    }
}
