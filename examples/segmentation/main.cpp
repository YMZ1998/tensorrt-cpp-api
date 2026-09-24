#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tensorrt_cpp_api/build_options.h"
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/engine_builder.h"
#include "tensorrt_cpp_api/tensor.h"

using namespace trtcpp;

class OctSegmentor {
public:
    bool load(const std::string &modelPath, const std::string &cacheDir) {
        BuildOptions options;
        options.precision = Precision::kFp16;
        options.engineCacheDir = cacheDir;

        auto result = EngineBuilder{}.buildAndLoad(modelPath, options);

        if (!result) {
            std::fprintf(stderr, "engine: %s\n", result.status().message().c_str());
            return false;
        }

        engine_ = std::make_unique<Engine>(std::move(result.value()));

        const auto inputs = engine_->inputNames();
        const auto outputs = engine_->outputNames();

        if (inputs.size() != 1 || outputs.empty()) {
            std::fprintf(stderr, "invalid model IO\n");
            return false;
        }

        inputName_ = inputs[0];
        outputName_ = outputs[0];

        auto shape = engine_->tensorShape(inputName_);
        if (!shape || shape->rank() != 4 || (*shape)[0] != 1 || (*shape)[1] != 1) {
            std::fprintf(stderr, "input must be [1,1,H,W]\n");
            return false;
        }

        inputH_ = static_cast<int>((*shape)[2]);
        inputW_ = static_cast<int>((*shape)[3]);

        return true;
    }

    bool infer(const cv::Mat &image, cv::Mat &mask, double *elapsedMs = nullptr) {
        if (image.empty() || image.type() != CV_8UC1)
            return false;

        auto input = preprocess(image);

        auto tensorResult = Tensor::allocate(DType::kFloat32, Shape{1, 1, inputH_, inputW_}, Device::kCuda);

        if (!tensorResult) {
            std::fprintf(stderr, "allocate input: %s\n", tensorResult.status().message().c_str());
            return false;
        }

        auto tensor = std::move(tensorResult.value());

        TensorView host{input.data(), DType::kFloat32, Shape{1, 1, inputH_, inputW_}, Device::kHost};

        if (auto s = tensor.copyFrom(host, stream_); !s) {
            std::fprintf(stderr, "upload: %s\n", s.message().c_str());
            return false;
        }

        const auto start = std::chrono::steady_clock::now();

        auto outputs = engine_->infer({{inputName_, tensor.view()}}, stream_);

        if (!outputs) {
            std::fprintf(stderr, "infer: %s\n", outputs.status().message().c_str());
            return false;
        }

        if (auto s = stream_.synchronize(); !s) {
            std::fprintf(stderr, "sync: %s\n", s.message().c_str());
            return false;
        }

        if (elapsedMs) {
            *elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }

        auto it = outputs->find(outputName_);
        if (it == outputs->end())
            return false;

        auto hostResult = it->second.toHost(stream_);
        if (!hostResult) {
            std::fprintf(stderr, "download: %s\n", hostResult.status().message().c_str());
            return false;
        }

        auto hostTensor = std::move(hostResult.value());

        if (hostTensor.shape().rank() != 4 || hostTensor.shape()[0] != 1 || hostTensor.dtype() != DType::kFloat32) {
            std::fprintf(stderr, "invalid output: %s\n", hostTensor.shape().toString().c_str());
            return false;
        }

        const int C = static_cast<int>(hostTensor.shape()[1]);
        const int H = static_cast<int>(hostTensor.shape()[2]);
        const int W = static_cast<int>(hostTensor.shape()[3]);

        auto logits = hostTensor.as<float>().value();

        mask = postprocess(logits, C, H, W);
        cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);

        return true;
    }

    int inputWidth() const { return inputW_; }

    int inputHeight() const { return inputH_; }

private:
    std::vector<float> preprocess(const cv::Mat &image) const {
        std::vector<float> dst(static_cast<size_t>(inputH_) * inputW_);
        const float sx = static_cast<float>(image.cols) / inputW_;
        const float sy = static_cast<float>(image.rows) / inputH_;
        const float cx = (image.cols - 1) * 0.5f;
        const float cy = (image.rows - 1) * 0.5f;
        const float r = std::min(image.cols, image.rows) * 0.5f;
        const float r2 = r * r;
        auto pixel = [&](int x, int y) {
            x = std::clamp(x, 0, image.cols - 1);
            y = std::clamp(y, 0, image.rows - 1);

            const float dx = x - cx;
            const float dy = y - cy;

            if (dx * dx + dy * dy > r2)
                return 0.0f;

            return static_cast<float>(image.at<uint8_t>(y, x));
        };

        for (int y = 0; y < inputH_; ++y) {
            const float fy = (y + 0.5f) * sy - 0.5f;

            const int y0 = static_cast<int>(std::floor(fy));
            const int y1 = y0 + 1;
            const float wy = fy - y0;

            for (int x = 0; x < inputW_; ++x) {
                const float fx = (x + 0.5f) * sx - 0.5f;

                const int x0 = static_cast<int>(std::floor(fx));
                const int x1 = x0 + 1;
                const float wx = fx - x0;

                const float a = pixel(x0, y0);
                const float b = pixel(x1, y0);
                const float c = pixel(x0, y1);
                const float d = pixel(x1, y1);

                const float top = a + (b - a) * wx;

                const float bottom = c + (d - c) * wx;

                dst[static_cast<size_t>(y) * inputW_ + x] = (top + (bottom - top) * wy) / 127.5f - 1.0f;
            }
        }

        return dst;
    }

    cv::Mat postprocess(std::span<const float> logits, int C, int H, int W) const {
        cv::Mat mask(H, W, CV_8UC1);

        const int plane = H * W;

        for (int p = 0; p < plane; ++p) {
            int cls = 0;
            float best = logits[p];

            for (int c = 1; c < C; ++c) {
                const float value = logits[static_cast<size_t>(c) * plane + p];

                if (value > best) {
                    best = value;
                    cls = c;
                }
            }

            mask.data[p] = C == 1 ? (best > 0.5f ? 255 : 0) : static_cast<uint8_t>(cls * 255 / (C - 1));
        }

        return mask;
    }

private:
    std::unique_ptr<Engine> engine_;
    Stream stream_;

    std::string inputName_;
    std::string outputName_;

    int inputH_ = 0;
    int inputW_ = 0;
};

int main() {
    const std::string root = "D:/Code/tensorrt-cpp-api";

    OctSegmentor segmentor;

    if (!segmentor.load(root + "/models/efficientnet_b1_best_model.onnx", root + "/models")) {
        return 1;
    }

    cv::Mat image = cv::imread(root + "/inputs/input.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::fprintf(stderr, "failed to read image\n");
        return 1;
    }

    // Warmup
    cv::Mat mask;
    for (int i = 0; i < 5; ++i) {
        if (!segmentor.infer(image, mask))
            return 1;
    }

    // Benchmark
    double ms = 0.0;

    if (!segmentor.infer(image, mask, &ms))
        return 1;

    const std::string output = root + "/inputs/segmentation.png";

    if (!cv::imwrite(output, mask)) {
        std::fprintf(stderr, "failed to save mask\n");
        return 1;
    }

    std::printf("input : %dx%d\n"
                "model : %dx%d\n"
                "infer : %.3f ms\n"
                "saved : %s\n",
                image.cols, image.rows, segmentor.inputWidth(), segmentor.inputHeight(), ms, output.c_str());

    return 0;
}