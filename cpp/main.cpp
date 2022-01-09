/*
MIT License

Copyright (c) 2022 mgmk2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <fstream>
#include <iostream>
#include <array>
#include <string>

#include <filesystem>

#include <cxxopts.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

struct ParseResult {
    bool isOk{ true };
    bool showHelp{ false };
    fs::path imageFile;
    fs::path labelFile;
    fs::path modelFile;
};

ParseResult parseArgments(int argc, char** argv)
{
    ParseResult parseResult;

    cxxopts::Options options("ONNXRuntimeResnet", "Resnet50 inference by ONNXRuntime.");
    options.add_options()
        ("i,input", "224 x 224 image's filename.", cxxopts::value<std::string>())
        ("r,resource", "resource directory with onnx model and imagenet labels.", cxxopts::value<std::string>())
        ("h,help", "show help.");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (cxxopts::OptionException& e) {
        std::cout << e.what() << std::endl;
        parseResult.isOk = false;
        return parseResult;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        parseResult.showHelp = true;
        return parseResult;
    }

    fs::path resourceDir = result["resource"].as<std::string>();
    if (!fs::exists(resourceDir)) {
        std::cout << resourceDir << " does not exists." << std::endl;
        parseResult.isOk = false;
        return parseResult;
    }

    parseResult.imageFile = result["input"].as<std::string>();
    parseResult.labelFile = resourceDir / "imagenet_classes.txt";
    parseResult.modelFile = resourceDir / "resnet50.onnx";

    for (const auto& f : { parseResult.imageFile, parseResult.labelFile, parseResult.modelFile }) {
        if (!fs::exists(f)) {
            std::cout << f << " does not exists." << std::endl;
            parseResult.isOk = false;
            return parseResult;
        }
        if (!fs::is_regular_file(f)) {
            std::cout << f << " is not a file." << std::endl;
            parseResult.isOk = false;
            return parseResult;
        }
    }
    parseResult.imageFile = fs::canonical(parseResult.imageFile);
    parseResult.labelFile = fs::canonical(parseResult.labelFile);
    parseResult.modelFile = fs::canonical(parseResult.modelFile);

    return parseResult;
}

std::vector<float> loadImage(const std::string& filename)
{
    auto image = cv::imread(filename);
    if (image.empty()) {
        return {};
    }

    // convert from BGR to RGB
    try {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    catch (cv::Exception&) {
        return {};
    }

    // reshape to 1D
    image = image.reshape(1, 1);

    // uint_8, [0, 255] -> float, [0, 1]
    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    // HWC -> CHW
    std::vector<float> output;
    for (size_t ch = 0; ch < 3; ++ch) {
        for (size_t i = ch; i < vec.size(); i += 3) {
            output.emplace_back(vec[i]);
        }
    }
    return output;
}

std::vector<std::string> loadLabels(const std::string& filename)
{
    std::vector<std::string> output;

    std::ifstream file(filename);
    if (file) {
        std::string s;
        while (std::getline(file, s)) {
            output.emplace_back(s);
        }
        file.close();
    }

    return output;
}

int main(int argc, char** argv)
{
    const auto parseResult = parseArgments(argc, argv);
    if (parseResult.showHelp) {
        return 0;
    }
    if (!parseResult.isOk) {
        return 1;
    }

    const std::string imageFile = parseResult.imageFile.string();
    const std::string labelFile = parseResult.labelFile.string();
    const std::wstring modelFile = parseResult.modelFile.wstring();

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;

    // load labels
    const std::vector<std::string> labels = loadLabels(labelFile);
    if (labels.empty()) {
        std::cout << "Failed to load labels: " << labelFile << std::endl;
        return 1;
    }

    // load image
    const std::vector<float> imageVec = loadImage(imageFile);
    if (imageVec.empty()) {
        std::cout << "Failed to load image: " << imageFile << std::endl;
        return 1;
    }
    if (imageVec.size() != numInputElements) {
        std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
        return 1;
    }

    Ort::Env env;

    Ort::SessionOptions sessionOptions;
    Ort::RunOptions runOptions;

    // create session
    // We should catch exception...
    Ort::Session session(nullptr);
    try {
        session = Ort::Session(env, modelFile.data(), sessionOptions);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // define I/O shape
    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    // define I/O array
    // We can use std::vector instead of std::array.
    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    // define I/O Tensor
    // It holds the array pointer internally.
    // DON'T delete array while the Tensor alive.
    // If use std::vector, DON'T reallocate memory after creating the Tensor.
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // copy image data to input array
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    // define I/O names
    const std::array<const char*, 1> inputNames = { "input0" };
    const std::array<const char*, 1> outputNames = { "output0" };

    /*
    // We can get I/O names from model data.
    // Be careful to the order of I/O.
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        inputNames.emplace_back(session.GetInputName(i, alloc));
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        outputNames.emplace_back(session.GetOutputName(i, alloc));
    }
    */

    // run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // sort results
    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // show Top5
    for (size_t i = 0; i < 5; ++i) {
        const auto& [index, value] = indexValuePairs[i];
        std::cout << i + 1 << ": " << labels[index] << " " << value << std::endl;
    }
}