#include <fstream>
#include <iostream>
#include <array>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

std::vector<float> loadImage(const std::string& filename)
{
    auto image = cv::imread(filename);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image = image.reshape(1, 1);
    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    // convert HWC -> CHW
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

int main()
{
    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;

    const std::vector<float> imageVec = loadImage("resource/dog_center.jpg");
    const std::vector<std::string> labels = loadLabels("resource/imagenet_classes.txt");

    Ort::Env env;

    Ort::SessionOptions sessionOptions;
    Ort::RunOptions runOptions;

    Ort::Session session(env, L"resource/resnet50.onnx", sessionOptions);

    const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { 1, numClasses };

    constexpr int64_t numInputElements = numChannels * height * width;

    std::array<float, numInputElements> input;
    std::array<float, numClasses> results;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    const std::array<const char*, 1> inputName = { "input0" };
    const std::array<const char*, 1> outputName = { "output0" };

    assert(imageVec.size() == numInputElements);
    std::copy(imageVec.begin(), imageVec.end(), input.begin());

    session.Run(runOptions, inputName.data(), &inputTensor, 1, outputName.data(), &outputTensor, 1);

    std::vector<std::pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results.size(); ++i) {
        indexValuePairs.emplace_back(i, results[i]);
    }
    std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    for (size_t i = 0; i < 5; ++i) {
        const auto [index, value] = indexValuePairs[i];
        std::cout
            << "index: " << index
            << " value: " << value
            << " label: " << labels[index]
            << std::endl;
    }
}