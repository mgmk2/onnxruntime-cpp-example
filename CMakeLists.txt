cmake_minimum_required(VERSION 3.15)
project(ORTResnet VERSION 0.1.0)

cmake_policy(SET CMP0048 NEW)

option(USE_DIST_ORT "Use distributed ONNXRuntime." OFF)
option(ORT_STATIC "Use ONNXRuntime static library." OFF)

cmake_policy(SET CMP0091 NEW)

message(STATUS "Use distributed ONNXRuntime: ${USE_DIST_ORT}")
if (USE_DIST_ORT)
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    if (ORT_STATIC)
        message(FATAL_ERROR "ONNXRuntime static library is not provided in the distributed.")
    endif()
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>DLL)
endif()

set(CMAKE_CXX_STANDARD 17)

# check ORT_ROOT existance
if (NOT ORT_ROOT)
    message(FATAL_ERROR "ORT_ROOT must be set")
endif()

get_filename_component(ORT_ROOT_ABS ${ORT_ROOT} ABSOLUTE)
if (EXISTS ${ORT_ROOT_ABS})
    message(STATUS "ONNXRuntime directory: ${ORT_ROOT}")
else()
    message(FATAL_ERROR "ORT_ROOT does not exists: ${ORT_ROOT}")
endif()

# define target
add_executable(ORTResnet cpp/main.cpp)

if(MSVC)
    target_compile_options(
        ORTResnet
    PRIVATE
        /source-charset:utf-8
    )
endif()

# ONNXRuntime
target_include_directories(
    ORTResnet
PRIVATE
    ${ORT_ROOT}/include
    ${ORT_ROOT}/include/onnxruntime/core/session
)

message(STATUS "ONNXRuntime STATIC: ${ORT_STATIC}")

if(ORT_STATIC)
    target_link_directories(
        ORTResnet
    PRIVATE
        ${ORT_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/static
        ${ORT_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/static/external
    )

    target_link_libraries(
        ORTResnet
    PRIVATE
        onnxruntime_session
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_util
        onnxruntime_framework
        onnxruntime_graph
        onnxruntime_mlas
        onnxruntime_common
        onnxruntime_flatbuffers
        onnx
        onnx_proto
        libprotobuf-lite$<$<CONFIG:Debug>:d>
        re2
        flatbuffers
        cpuinfo
        clog
        
        $<$<AND:$<PLATFORM_ID:Windows>,$<CONFIG:Debug>>:Dbghelp>
    )

else()
    target_link_directories(
        ORTResnet
    PRIVATE
        ${ORT_ROOT}/lib
        ${ORT_ROOT}/lib/$<IF:$<CONFIG:Debug>,Debug,Release>/shared
    )
    
    target_link_libraries(
        ORTResnet
    PRIVATE
        onnxruntime
    )
endif()

# cxxopts
target_include_directories(
    ORTResnet
PRIVATE
    external/cxxopts/include
)

# OpenCV
set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

target_include_directories(ORTResnet PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ORTResnet PRIVATE ${OpenCV_LIBS})