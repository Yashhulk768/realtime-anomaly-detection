cmake_minimum_required(VERSION 3.18)
project(anomaly_detect_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # optimised build for realistic timings
endif()

# Prebuilt ONNX Runtime release, unpacked into cpp/third_party/onnxruntime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Path to the unpacked ONNX Runtime release")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_library(ONNXRUNTIME_LIB onnxruntime
             PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(anomaly_detect src/main.cpp)
target_include_directories(anomaly_detect PRIVATE
                           "${ONNXRUNTIME_DIR}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(anomaly_detect PRIVATE "${ONNXRUNTIME_LIB}" ${OpenCV_LIBS})
target_compile_options(anomaly_detect PRIVATE -Wall -Wextra)

# Let the program find libonnxruntime.so at run time without extra setup
set_target_properties(anomaly_detect PROPERTIES BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
