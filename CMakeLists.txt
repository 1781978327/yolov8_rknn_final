cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov8_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# =====================================================================
# 0. BYTETracker 依赖 Eigen3
# =====================================================================
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3 /usr/local/include NO_DEFAULT_PATH)
if(NOT EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core)
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev: sudo apt install libeigen3-dev")
endif()
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

# =====================================================================
# 1. RKNN API (bundled runtime library)
# =====================================================================
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)

# =====================================================================
# 2. OpenCV 4.10.0（编译安装版，硬件加速）
#    查找 /usr/local/lib/cmake/opencv4/OpenCVConfig.cmake
# =====================================================================
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio imgcodecs)
if(OpenCV_FOUND)
  message(STATUS "OpenCV: ${OpenCV_VERSION} (hardware-accelerated) @ ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(OpenCV_LIBS ${OpenCV_LIBS} ${OpenCV_LIBS})
  set(OpenCV_INSTALL_LIBS "")
else()
  message(FATAL_ERROR "OpenCV 4.10.0 not found at /usr/local. Run: cmake ../.. -DOpenCV_DIR=/usr/local/lib/cmake/opencv4")
endif()

# =====================================================================
# 3. RGA 硬件图像加速（系统库）
# =====================================================================
set(RGA_LIB /usr/lib/aarch64-linux-gnu/librga.so)
include_directories(/usr/include/rga)
message(STATUS "RGA: using system librga.so from /usr/lib/aarch64-linux-gnu")

# =====================================================================
# 4. FFmpeg + RKMPP 硬件编解码
#    优先使用 -DFFMPEG_RKMPP_ROOT=/path/to/install 指定的位置，
#    若未指定则尝试从其他项目构建目录复用 FFmpeg 60 RKMPP 库。
# =====================================================================
set(FFMPEG_RKMPP_FOUND 0)

if(FFMPEG_RKMPP_ROOT)
  set(FFMPEG_RKMPP_INCLUDE "${FFMPEG_RKMPP_ROOT}/include")
  set(FFMPEG_RKMPP_LIB     "${FFMPEG_RKMPP_ROOT}/lib")
  if(NOT FFMPEG_RKMPP_RPATH)
    set(FFMPEG_RKMPP_RPATH "${FFMPEG_RKMPP_LIB}")
  endif()
  if(EXISTS "${FFMPEG_RKMPP_INCLUDE}/libavcodec/avcodec.h")
    set(FFMPEG_RKMPP_FOUND 1)
    message(STATUS "FFmpeg RKMPP: using ${FFMPEG_RKMPP_ROOT} (user-specified)")
  endif()
endif()

# 如果没有指定 FFMPEG_RKMPP_ROOT，尝试从其他项目构建目录查找 FFmpeg 60 库
if(NOT FFMPEG_RKMPP_FOUND)
  # 尝试从同级的 cpp-inference-main 项目复用 FFmpeg 60 RKMPP 库
  set(FFMPEG_FALLBACK_DIR /home/orangepi/Desktop/web/yolov8-rk3588-cpp-inference-main/build/lib)
  if(EXISTS "${FFMPEG_FALLBACK_DIR}/libavcodec.so.60")
    set(FFMPEG_RKMPP_ROOT "${FFMPEG_FALLBACK_DIR}")
    set(FFMPEG_RKMPP_INCLUDE "${FFMPEG_FALLBACK_DIR}")
    set(FFMPEG_RKMPP_LIB   "${FFMPEG_FALLBACK_DIR}")
    set(FFMPEG_RKMPP_RPATH "${FFMPEG_FALLBACK_DIR}")
    set(FFMPEG_RKMPP_FOUND 1)
    message(STATUS "FFmpeg RKMPP: using fallback FFmpeg 60 from ${FFMPEG_FALLBACK_DIR}")
  endif()
endif()

# 如果还是没有 FFmpeg RKMPP，检查系统 FFmpeg 版本
if(NOT FFMPEG_RKMPP_FOUND)
  # 检查系统 FFmpeg 版本
  find_path(AVCODEC_INCLUDE libavcodec/avcodec.h PATHS /usr/include)
  if(AVCODEC_INCLUDE)
    file(READ "${AVCODEC_INCLUDE}/libavcodec/version.h" AVCODEC_VERSION_CONTENT)
    string(REGEX MATCH "#define LIBAVCODEC_VERSION_MAJOR ([0-9]+)" _ "${AVCODEC_VERSION_CONTENT}")
    set(AVCODEC_VER_MAJOR "${CMAKE_MATCH_1}")
    if(AVCODEC_VER_MAJOR EQUAL 60)
      set(FFMPEG_RKMPP_INCLUDE "/usr/include")
      set(FFMPEG_RKMPP_LIB   "/usr/lib/aarch64-linux-gnu")
      set(FFMPEG_RKMPP_RPATH "/usr/lib/aarch64-linux-gnu")
      set(FFMPEG_RKMPP_FOUND 1)
      message(STATUS "FFmpeg RKMPP: using system FFmpeg 60 from /usr/lib/aarch64-linux-gnu")
    else()
      message(WARNING "System FFmpeg version is ${AVCODEC_VER_MAJOR} (expected 60). RTSP feature will be DISABLED.")
      message(WARNING "To enable RTSP: compile FFmpeg 60 with --enable-rkmpp and set -DFFMPEG_RKMPP_ROOT=/path/to/install")
    endif()
  endif()
endif()

# 配置 RTSP 功能 - 使用 /usr/local/ffmpeg FFmpeg 60
if(EXISTS "/usr/local/ffmpeg/lib/libavcodec.so.60")
  set(FFMPEG_RKMPP_ROOT "/usr/local/ffmpeg")
  set(FFMPEG_RKMPP_INCLUDE "${FFMPEG_RKMPP_ROOT}/include")
  set(FFMPEG_RKMPP_LIB   "${FFMPEG_RKMPP_ROOT}/lib")
  set(FFMPEG_RKMPP_RPATH "${FFMPEG_RKMPP_ROOT}/lib")
  set(FFMPEG_RKMPP_FOUND 1)
  message(STATUS "FFmpeg RKMPP: using /usr/local/ffmpeg FFmpeg 60")
else()
  set(FFMPEG_RKMPP_FOUND 0)
endif()

if(FFMPEG_RKMPP_FOUND)
  add_compile_definitions(USE_RTSP_MPP)
  include_directories(${FFMPEG_RKMPP_INCLUDE})
  link_directories(${FFMPEG_RKMPP_LIB})
  set(RTSP_MPP_SRCS src/rtsp_mpp_sender.cpp)
  set(VIDEO_RKMPP_READER_SRCS src/ffmpeg_rkmpp_reader.cpp)
  set(RTSP_MPP_LIBS -lavformat -lavcodec -lavutil -lswscale -lswresample -lavdevice -lz -lpthread -llzma -lm -ldrm -lrockchip_mpp)
  set(RTSP_MPP_BUILD_RPATH "${FFMPEG_RKMPP_RPATH}")
else()
  set(RTSP_MPP_SRCS "")
  set(VIDEO_RKMPP_READER_SRCS "")
  set(RTSP_MPP_LIBS "")
  set(RTSP_MPP_BUILD_RPATH "")
endif()

# =====================================================================
# 5. BYTETracker 跟踪算法库
# =====================================================================
set(BYTETRACK_DIR ${CMAKE_SOURCE_DIR}/bytetrack)
aux_source_directory(${BYTETRACK_DIR}/src BYTETRACK_SRC)
add_library(bytetrack STATIC ${BYTETRACK_SRC})
target_include_directories(bytetrack PRIVATE
    ${BYTETRACK_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(bytetrack ${OpenCV_LIBS})

# =====================================================================
# 5.1 DeepSORT 跟踪算法库
# =====================================================================
set(DEEPSORT_DIR ${CMAKE_SOURCE_DIR}/deepsort)
aux_source_directory(${DEEPSORT_DIR}/src DEEPSORT_SRC)
add_library(deepsort STATIC ${DEEPSORT_SRC})
target_include_directories(deepsort PRIVATE
    ${DEEPSORT_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(deepsort
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${OpenCV_LIBS}
)

# =====================================================================
# 6. 主程序构建
# =====================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/bytetrack/include
    ${CMAKE_SOURCE_DIR}/deepsort/include
)

add_executable(rknn_yolov8_demo
  src/main.cc
  src/postprocess.cc
  src/rk_common.cc
  src/algo_reporter.cpp
  src/v4l2_dmabuf_capture.cpp
  ${RTSP_MPP_SRCS}
)

# 使用 OpenCV 4 模块化 target（兼容 cmake find_package）
# 同时保留 OpenCV_LIBS 变量供兼容
if(OpenCV_FOUND)
  target_link_libraries(rknn_yolov8_demo
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_imgcodecs
    ${RTSP_MPP_LIBS}
    bytetrack
    deepsort
  )
else()
  target_link_libraries(rknn_yolov8_demo
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    ${OpenCV_LIBS}
    ${RTSP_MPP_LIBS}
    deepsort
  )
endif()

# 设置 RPATH（运行时库搜索路径）
if(RTSP_MPP_BUILD_RPATH)
  set_target_properties(rknn_yolov8_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/lib;${RTSP_MPP_BUILD_RPATH}"
    INSTALL_RPATH "$ORIGIN/lib;${RTSP_MPP_BUILD_RPATH}"
  )
  target_link_options(rknn_yolov8_demo PRIVATE -Wl,--disable-new-dtags)

  # 复制 FFmpeg RKMPP 动态库到 build/lib
  add_custom_command(TARGET rknn_yolov8_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:rknn_yolov8_demo>/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${FFMPEG_RKMPP_LIB}/libavformat.so.60"
      "${FFMPEG_RKMPP_LIB}/libavcodec.so.60"
      "${FFMPEG_RKMPP_LIB}/libavutil.so.58"
      "${FFMPEG_RKMPP_LIB}/libswscale.so.7"
      "${FFMPEG_RKMPP_LIB}/libswresample.so.4"
      "$<TARGET_FILE_DIR:rknn_yolov8_demo>/lib"
    VERBATIM
  )
endif()

# =====================================================================
# 7. HTTP 控制服务（可选）
#    提供 HTTP API + Web 控制面板，支持与 Spring Boot 对接
# =====================================================================
set(HTTP_CTRL_SOURCES
    src/main_http_ctrl.cc
    src/postprocess.cc
    src/rk_common.cc
    src/v4l2_dmabuf_capture.cpp
    ${VIDEO_RKMPP_READER_SRCS}
    ${RTSP_MPP_SRCS}
)

add_executable(rknn_http_ctrl ${HTTP_CTRL_SOURCES})

target_include_directories(rknn_http_ctrl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/deepsort/include
)

target_link_libraries(rknn_http_ctrl
    ${RKNN_RT_LIB}
    ${RGA_LIB}
    opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_imgcodecs
    ${RTSP_MPP_LIBS}
    pthread
    bytetrack
    deepsort
)

# =====================================================================
# 8. 安装规则
# =====================================================================
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov8_demo rknn_http_ctrl DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS /usr/lib/aarch64-linux-gnu/librga.so DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
