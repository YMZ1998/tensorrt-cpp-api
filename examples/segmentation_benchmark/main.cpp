// Benchmark OCT-style segmentation prediction latency.
//
// Usage:
//   segmentation_benchmark [iterations] [warmup] [model.onnx|engine] [image]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../common/image_io.h"

using namespace trtcpp;

namespace {

float maskedGrayAt(const examples::Image &img, int x, int y) {
    x = std::clamp(x, 0, img.width - 1);
    y = std::clamp(y, 0, img.height - 1);

    const float cx = (static_cast<float>(img.width) - 1.0f) * 0.5f;
    const float cy = (static_cast<float>(img.height) - 1.0f) * 0.5f;
    const float radius = static_cast<float>(std::min(img.width, img.height)) * 0.5f;
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    if (dx * dx + dy * dy > radius * radius) {
        return 0.0f;
    }

    const auto *px = &img.data[(static_cast<std::size_t>(y) * img.width + x) * 3];
    return 0.299f * static_cast<float>(px[0]) + 0.587f * static_cast<float>(px[1]) + 0.114f * static_cast<float>(px[2]);
}

std::vector<float> preprocessOctMinusOneToOne(const examples::Image &img, int outH, int outW) {
    std::vector<float> chw(static_cast<std::size_t>(outH) * outW);
    const float scaleY = static_cast<float>(img.height) / static_cast<float>(outH);
    const float scaleX = static_cast<float>(img.width) / static_cast<float>(outW);

    for (int y = 0; y < outH; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = y0 + 1;
        const float wy = sy - static_cast<float>(y0);
        for (int x = 0; x < outW; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            const int x0 = static_cast<int>(std::floor(sx));
            const int x1 = x0 + 1;
            const float wx = sx - static_cast<float>(x0);

            const float v00 = maskedGrayAt(img, x0, y0);
            const float v01 = maskedGrayAt(img, x1, y0);
            const float v10 = maskedGrayAt(img, x0, y1);
            const float v11 = maskedGrayAt(img, x1, y1);
            const float top = v00 + (v01 - v00) * wx;
            const float bottom = v10 + (v11 - v10) * wx;
            const float pixel = top + (bottom - top) * wy;

            chw[static_cast<std::size_t>(y) * outW + x] = pixel / 127.5f - 1.0f;
        }
    }

    return chw;
}

double percentile(std::vector<double> values, double pct) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = (pct / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(rank));
    const auto hi = static_cast<std::size_t>(std::ceil(rank));
    if (lo == hi) {
        return values[lo];
    }
    const double w = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - w) + values[hi] * w;
}

int parsePositive(const char *text, int fallback) {
    if (text == nullptr) {
        return fallback;
    }
    const int value = std::atoi(text);
    return value > 0 ? value : fallback;
}

} // namespace

int main(int argc, char **argv) {
    const std::string root = "D:/Code/tensorrt-cpp-api";
    const int iterations = argc > 1 ? parsePositive(argv[1], 30) : 30;
    const int warmup = argc > 2 ? std::max(0, std::atoi(argv[2])) : 5;
    const std::string modelPath = argc > 3 ? argv[3] : root + "/models/efficientnet_b1_best_model.onnx";
    const std::string imagePath = argc > 4 ? argv[4] : root + "/inputs/input.png";

    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = root + "/models";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }

    const auto inputNames = engine->inputNames();
    if (inputNames.size() != 1) {
        std::fprintf(stderr, "segmentation benchmark expects exactly one input\n");
        return 1;
    }

    const std::string inName = inputNames.front();
    auto inShapeResult = engine->tensorShape(inName);
    if (!inShapeResult || inShapeResult->rank() != 4 || (*inShapeResult)[0] != 1 || (*inShapeResult)[1] != 1 ||
        (*inShapeResult)[2] <= 0 || (*inShapeResult)[3] <= 0) {
        std::fprintf(stderr, "segmentation benchmark expects input shape [1,1,H,W]\n");
        return 1;
    }

    const auto inShape = inShapeResult.value();
    const int inH = static_cast<int>(inShape[2]);
    const int inW = static_cast<int>(inShape[3]);

    examples::Image img = examples::decodeImage(imagePath);
    if (img.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    Stream stream;
    auto inputHost = preprocessOctMinusOneToOne(img, inH, inW);
    auto dstResult = Tensor::allocate(DType::kFloat32, Shape{1, 1, inH, inW}, Device::kCuda);
    if (!dstResult) {
        std::fprintf(stderr, "allocate input: %s\n", dstResult.status().message().c_str());
        return 1;
    }
    auto dst = std::move(dstResult.value());
    TensorView hostInput{inputHost.data(), DType::kFloat32, Shape{1, 1, inH, inW}, Device::kHost};
    if (auto s = dst.copyFrom(hostInput, stream); !s) {
        std::fprintf(stderr, "upload input: %s\n", s.message().c_str());
        return 1;
    }
    if (auto s = stream.synchronize(); !s) {
        std::fprintf(stderr, "input synchronize: %s\n", s.message().c_str());
        return 1;
    }

    std::printf("model:      %s\n", modelPath.c_str());
    std::printf("image:      %s\n", imagePath.c_str());
    std::printf("input:      [1,1,%d,%d]\n", inH, inW);
    std::printf("warmup:     %d\n", warmup);
    std::printf("iterations: %d\n", iterations);

    auto runOnce = [&]() -> Result<double> {
        const auto start = std::chrono::steady_clock::now();
        auto outputs = engine->infer({{inName, dst.view()}}, stream);
        if (!outputs) {
            return outputs.status();
        }
        if (auto s = stream.synchronize(); !s) {
            return s;
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    for (int i = 0; i < warmup; ++i) {
        auto elapsed = runOnce();
        if (!elapsed) {
            std::fprintf(stderr, "warmup infer: %s\n", elapsed.status().message().c_str());
            return 1;
        }
    }

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        auto elapsed = runOnce();
        if (!elapsed) {
            std::fprintf(stderr, "infer: %s\n", elapsed.status().message().c_str());
            return 1;
        }
        times.push_back(elapsed.value());
        std::printf("run %3d/%d: %.3f ms\n", i + 1, iterations, elapsed.value());
    }

    const auto minMax = std::minmax_element(times.begin(), times.end());
    const double avg = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());

    std::printf("\nprediction time summary:\n");
    std::printf("  avg: %.3f ms\n", avg);
    std::printf("  min: %.3f ms\n", *minMax.first);
    std::printf("  p50: %.3f ms\n", percentile(times, 50.0));
    std::printf("  p90: %.3f ms\n", percentile(times, 90.0));
    std::printf("  max: %.3f ms\n", *minMax.second);

    return 0;
}
