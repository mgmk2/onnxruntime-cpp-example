# onnxruntime-cpp-example
This repo is a project for a ResNet50 inference application using [ONNXRuntime][ort] in C++.

Currently, I build and test on Windows10 with Visual Studio 2019 only.
All resources (build-system, dependencies and etc...) are cross-platform, so maybe you can build the application on other environment.

[ort]: https://onnxruntime.ai/

## Requirements
### C++
* C++17
* [cxxopts][cxxoptsurl] (submodule)
* [OpenCV][ocvurl] (you have to install manually)

[cxxoptsurl]: https://github.com/jarro2783/cxxopts
[ocvurl]: https://opencv.org/

### Python
* Python3
* [PyTorch][pturl]
* [Pillow][pilurl]
* [Numpy][npurl]

[pturl]: https://pytorch.org/
[pilurl]: https://pillow.readthedocs.io/en/stable/
[npurl]: https://numpy.org/

## Get Started
If you don't have ONNXRuntime, you have two options:
1. Download the distribution package
2. Build from sources

If you download the distribution package, skip the next section about building ONNXRuntime.

NOTE: If you want to link ONNXRuntime statically, you have to build ONNXRuntime from sources.

### Build ONNXRuntime
You can build ONNXRuntime from sources like below:

```
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# if win
build.bat --config Debug --build_shared_lib --parallel
build.bat --config Release --build_shared_lib --parallel

# else
build.sh --config Debug --build_shared_lib --parallel
build.sh --config Release --build_shared_lib --parallel
```

After build and test, place libraries like below:

```
${ORT_ROOT}
├─include # headers
└─lib
    ├─Debug # DO NOT forget pdb files...
    │  ├─shared
    │  │   ├─onnxruntime.dll
    │  │   └─onnxruntime.lib
    │  └─static
    │      ├─onnxruntime_common.lib
    │      ├─onnxruntime_flatbuffers.lib
    │      ├─onnxruntime_framework.lib
    │      ├─onnxruntime_graph.lib
    │      ├─onnxruntime_mlas.lib
    │      ├─onnxruntime_optimizer.lib
    │      ├─onnxruntime_providers.lib
    │      ├─onnxruntime_session.lib
    │      ├─onnxruntime_util.lib
    │      └─external (dependencies)
    │          ├─clog.lib
    │          ├─cpuinfo.lib
    │          ├─flatbuffers.lib
    │          ├─libprotobuf-lited.lib
    │          ├─onnx.lib
    │          ├─onnx_proto.lib
    │          └─re2.lib
    ├─Release
    │  :
    :
```

### Download Resources
Before you build the application, you have to output resources like ResNet50 model of ONNX format, imagenet labels and a test image.

To do this, run `python/output_resource.py` like below:

```
python python/output_resource.py -o resource
```

After run above command, you can see a directory named `resource`.

### Build Application
You can build the application by cmake like below:

```
cmake -B build -DORT_STATIC=OFF -DUSE_DIST_ORT=OFF -DORT_ROOT=path/to/ORT_ROOT
cmake --build build --config Debug
```

### Run Application
After build succeeded, you can run application and see inference results.

```
build/Debug/ORTResnet.exe -i resource/dog_input.png -r reource
```

To validate results, run `python/check_inference_result.py` and compare to application results.

```
python python/check_inference_result.py -i resource/dog_input.png -l resource/imagenet_classes.txt
```