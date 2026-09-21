// Semantic segmentation with an OCT-style single-channel TensorRT model.
//
// Pipeline: decode (stb RGB) -> grayscale + circular ROI -> resize -> normalize to [-1, 1] ->
// NCHW [1,1,H,W] -> infer -> per-pixel prediction over [1,C,H,W] -> single-channel mask -> write.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

} // namespace

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const std::string root = "D:/Code/tensorrt-cpp-api";

    const std::string modelPath = root + "/models/efficientnet_b1_best_model.onnx";
    const std::string imagePath = root + "/inputs/input.png";
    const std::string outPath = root + "/inputs/segmentation.jpg";

    BuildOptions bo;
    bo.precision = Precision::kFp16;
    bo.engineCacheDir = root + "/models";
    auto engine = EngineBuilder{}.buildAndLoad(modelPath, bo);
    if (!engine) {
        std::fprintf(stderr, "engine: %s\n", engine.status().message().c_str());
        return 1;
    }
    const auto inputNames = engine->inputNames();
    const auto outputNames = engine->outputNames();
    if (inputNames.size() != 1) {
        std::fprintf(stderr, "segmentation expects exactly one input\n");
        return 1;
    }
    if (outputNames.empty()) {
        std::fprintf(stderr, "segmentation expects at least one output\n");
        return 1;
    }

    const std::string inName = inputNames.front();
    std::string outName;
    std::string fallbackOutName;
    for (const std::string &name : outputNames) {
        auto shapeResult = engine->tensorShape(name);
        auto dtypeResult = engine->tensorDType(name);
        if (!shapeResult || !dtypeResult) {
            continue;
        }
        const Shape shape = shapeResult.value();
        const bool isSegmentationLogits =
            *dtypeResult == DType::kFloat32 && shape.rank() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0;
        if (!isSegmentationLogits) {
            continue;
        }
        if (shape[1] == 1) {
            outName = name;
            break;
        }
        if (fallbackOutName.empty()) {
            fallbackOutName = name;
        }
        if (name == "out" || name.find("out") != std::string::npos) {
            outName = name;
            break;
        }
    }
    if (outName.empty()) {
        outName = fallbackOutName;
    }
    if (outName.empty()) {
        std::fprintf(stderr, "could not find a float32 segmentation output [1,C,H,W]\n");
        for (const std::string &name : outputNames) {
            auto shapeResult = engine->tensorShape(name);
            auto dtypeResult = engine->tensorDType(name);
            if (shapeResult && dtypeResult) {
                const std::string dtype = std::string(toString(dtypeResult.value()));
                std::fprintf(stderr, "  output: %s shape=%s dtype=%s\n", name.c_str(), shapeResult->toString().c_str(),
                             dtype.c_str());
            }
        }
        return 1;
    }

    auto inShapeResult = engine->tensorShape(inName);
    if (!inShapeResult || inShapeResult->rank() != 4 || (*inShapeResult)[0] != 1 || (*inShapeResult)[1] != 1 ||
        (*inShapeResult)[2] <= 0 || (*inShapeResult)[3] <= 0) {
        std::fprintf(stderr, "segmentation expects input shape [1,1,H,W]\n");
        return 1;
    }
    const auto inShape = inShapeResult.value(); // [1,1,H,W]
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

    const auto inferStart = std::chrono::steady_clock::now();
    auto outputs = engine->infer({{inName, dst.view()}}, stream);
    if (!outputs) {
        std::fprintf(stderr, "infer: %s\n", outputs.status().message().c_str());
        return 1;
    }
    if (auto s = stream.synchronize(); !s) {
        std::fprintf(stderr, "infer synchronize: %s\n", s.message().c_str());
        return 1;
    }
    const auto inferEnd = std::chrono::steady_clock::now();
    const double inferMs = std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
    auto outIt = outputs->find(outName);
    if (outIt == outputs->end()) {
        std::fprintf(stderr, "infer did not return selected output: %s\n", outName.c_str());
        return 1;
    }
    auto hostResult = outIt->second.toHost(stream);
    if (!hostResult) {
        std::fprintf(stderr, "copy output to host: %s\n", hostResult.status().message().c_str());
        return 1;
    }
    auto host = std::move(hostResult.value());
    if (host.shape().rank() != 4 || host.shape()[0] != 1 || host.shape()[1] <= 0 || host.shape()[2] <= 0 || host.shape()[3] <= 0 ||
        host.dtype() != DType::kFloat32) {
        std::fprintf(stderr, "segmentation output %s must be float32 with shape [1,C,H,W], got shape %s\n", outName.c_str(),
                     host.shape().toString().c_str());
        return 1;
    }
    auto logits = host.as<float>().value(); // [1,C,H,W], channels-first
    const int C = static_cast<int>(host.shape()[1]);
    const int H = static_cast<int>(host.shape()[2]);
    const int W = static_cast<int>(host.shape()[3]);
    const int plane = H * W;

    std::vector<std::uint8_t> classMap(static_cast<std::size_t>(plane));
    const int reportedClasses = C == 1 ? 2 : C;
    std::vector<int> hist(static_cast<std::size_t>(reportedClasses), 0);
    if (C == 1) {
        for (int p = 0; p < plane; ++p) {
            const int cls = logits[static_cast<std::size_t>(p)] > 0.5f ? 1 : 0;
            classMap[static_cast<std::size_t>(p)] = static_cast<std::uint8_t>(cls);
            ++hist[static_cast<std::size_t>(cls)];
        }
    } else {
        for (int p = 0; p < plane; ++p) {
            int best = 0;
            float bestVal = logits[p];
            for (int c = 1; c < C; ++c) {
                const float v = logits[static_cast<std::size_t>(c) * plane + p];
                if (v > bestVal) {
                    bestVal = v;
                    best = c;
                }
            }
            classMap[static_cast<std::size_t>(p)] = static_cast<std::uint8_t>(best);
            ++hist[static_cast<std::size_t>(best)];
        }
    }

    examples::Image prediction;
    prediction.width = img.width;
    prediction.height = img.height;
    prediction.channels = 3;
    prediction.data.resize(static_cast<std::size_t>(img.width) * img.height * 3);
    for (int y = 0; y < img.height; ++y) {
        const int sy = y * H / img.height;
        for (int x = 0; x < img.width; ++x) {
            const int sx = x * W / img.width;
            const std::uint8_t cls = classMap[static_cast<std::size_t>(sy) * W + sx];
            const std::uint8_t value =
                C == 1 ? static_cast<std::uint8_t>(cls * 255) : static_cast<std::uint8_t>(cls * 255 / (C - 1));
            auto *px = &prediction.data[(static_cast<std::size_t>(y) * img.width + x) * 3];
            px[0] = value;
            px[1] = value;
            px[2] = value;
        }
    }
    examples::writeJpg(outPath, prediction);

    std::printf("preprocessed image as [1,1,%d,%d] with normalized = pixel / 127.5 - 1.0\n", inH, inW);
    std::printf("prediction time: %.3f ms\n", inferMs);
    std::printf("segmented %dx%d image into single-channel prediction; present classes:\n", img.width, img.height);
    for (int c = 0; c < reportedClasses; ++c) {
        if (hist[static_cast<std::size_t>(c)] > 0) {
            std::printf("  class %2d : %5.1f%% of pixels\n", c, 100.0 * hist[static_cast<std::size_t>(c)] / plane);
        }
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
}
