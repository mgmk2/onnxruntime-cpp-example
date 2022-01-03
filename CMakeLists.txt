cmake_minimum_required(VERSION 3.15)
project(ONNXRuntimeResnet VERSION 0.1.0)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_STATIC ON)

add_executable(ONNXRuntimeResnet cpp/main.cpp)

target_include_directories(
    ONNXRuntimeResnet
PRIVATE
    $ENV{ONNXRUNTIME_ROOT}/include
)

if(ONNXRUNTIME_STATIC)
    target_link_directories(
        ONNXRuntimeResnet
    PRIVATE
        $ENV{ONNXRUNTIME_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/static
        $ENV{ONNXRUNTIME_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/static/external
    )

    target_link_libraries(
        ONNXRuntimeResnet
    PRIVATE
        onnxruntime_session.lib
        onnxruntime_optimizer.lib
        onnxruntime_providers.lib
        onnxruntime_util.lib
        onnxruntime_framework.lib
        onnxruntime_graph.lib
        onnxruntime_mlas.lib
        onnxruntime_common.lib
        onnxruntime_flatbuffers.lib
        onnx.lib
        onnx_proto.lib
        libprotobuf-lite$<$<CONFIG:Debug>:d>.lib
        re2.lib
        flatbuffers.lib
        cpuinfo.lib
        clog.lib
        
        $<$<CONFIG:Debug>:Dbghelp.lib>
    )

else()
    target_link_directories(
        ONNXRuntimeResnet
    PRIVATE
        $ENV{ONNXRUNTIME_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/shared
    )
    
    target_link_libraries(
        ONNXRuntimeResnet
    PRIVATE
        onnxruntime.lib
        $<$<CONFIG:Debug>:Dbghelp.lib>
    )
endif()

# OpenCV
set(OpenCV_STATIC ON)
set(CMAKE_MODULE_PATH $ENV{OPENCV_ROOT})
find_package(OpenCV REQUIRED)

target_include_directories(ONNXRuntimeResnet PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ONNXRuntimeResnet PRIVATE ${OpenCV_LIBS})