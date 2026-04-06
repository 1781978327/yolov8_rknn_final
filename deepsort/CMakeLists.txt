cmake_minimum_required(VERSION 3.0.0)

set(PROJECT_DEEPSORT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Eigen3
if(NOT EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3 /usr/local/include NO_DEFAULT_PATH)
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core)
endif()
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev.")
endif()

# RGA 路径
set(RGA_PATH ${PROJECT_SOURCE_DIR}/include/3rdparty/rga/RK3588)

set(include_libs
    ${PROJECT_DEEPSORT_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/3rdparty/librknn_api/include
    ${PROJECT_SOURCE_DIR}/include/3rdparty/rga/include
    ${RGA_PATH}/include
    ${EIGEN3_INCLUDE_DIR}
)

# RKNN / RGA
find_library(RKNN_LIB rknnrt)
find_library(RGA_LIB rga)
set(dynamic_libs ${RKNN_LIB} ${RGA_LIB})

include_directories(${include_libs})

aux_source_directory(${PROJECT_DEEPSORT_DIR}/src DEEPSORT_SRC)

add_library(deepsort STATIC ${DEEPSORT_SRC})

find_package(OpenCV COMPONENTS core imgproc REQUIRED)
target_include_directories(deepsort PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(deepsort ${OpenCV_LIBS})
target_link_libraries(deepsort ${dynamic_libs})

set_target_properties(deepsort PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
