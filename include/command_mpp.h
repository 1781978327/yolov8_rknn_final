#ifndef COMMAND_MPP
#define COMMAND_MPP

#include <string>
#include <iostream>

class Command {
private:
    std::string url;
    std::string codec_name;
    int use_hw;
    std::string preset, tune, profile;
    int fps;
    std::string protocol;
    std::string capture_name;
    std::string trans_protocol;
    int width;
    int height;

public:
    Command()
        : use_hw(0), fps(30), preset("ultrafast"), tune("zerolatency"), profile("high"),
          protocol("rtsp"), capture_name("x11grab"), trans_protocol("tcp"), width(1440), height(900) {}

    void set_url(const char* u) { url = std::string(u); }
    void set_fps(int f) { fps = f; }
    void set_protocol(const char* pl) { protocol = std::string(pl); }
    void set_trans_protocol(const char* tp) { trans_protocol = std::string(tp); }
    void set_width(const char* w) { width = atoi(w); }
    void set_height(const char* h) { height = atoi(h); }
    void set_width(int w) { width = w; }
    void set_height(int h) { height = h; }
    void set_use_hw(int hw) { use_hw = hw; }

    const char* get_url() const { return url.c_str(); }
    int get_fps() const { return fps; }
    const char* get_protocol() const { return protocol.c_str(); }
    const char* get_trans_protocol() const { return trans_protocol.c_str(); }
    int get_width() const { return width; }
    int get_height() const { return height; }
    int get_use_hw() const { return use_hw; }
};

#endif
