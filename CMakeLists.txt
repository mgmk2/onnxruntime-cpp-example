# ORTResnet Build settings
# Three options are available.
#  1. ORT_ROOT    : (REQUIRED) The path of ONNXRuntime root directory.
#  2. USE_DIST_ORT: Use distributed ONNXRuntime binary instead of build from sources.
#  3. ORT_STATIC  : Link ONNXRuntime statically.
#
# ===================================================================================
#
# MIT License
# 
# Copyright (c) 2022 mgmk2
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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