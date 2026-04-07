#ifndef HTTP_CTRL_WEB_UTILS_H
#define HTTP_CTRL_WEB_UTILS_H

#include <string>

std::string get_timestamp();
std::string json_escape(const std::string& input);
std::string build_json_response(const std::string& status, const std::string& message = "");
bool parse_json_body(const std::string& body, std::string& path, bool& loop);
std::string build_html_page();
void send_response(int client_fd, const std::string& content, const std::string& content_type, int status_code = 200);

#endif
