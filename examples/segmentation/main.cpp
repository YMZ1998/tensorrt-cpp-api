// Semantic segmentation with DeepLabV3-MobileNetV3 + tensorrt_cpp_api.
//
// Pipeline: decode (stb) -> upload -> resize/normalize/NCHW (preproc) -> infer ->
// per-pixel argmax over the [1, 21, H, W] class logits -> colorize (Pascal VOC palette) ->
// blend with the input -> write.
//
// Export the model with torchvision (fixed 520x520):
//   deeplabv3_mobilenet_v3_large(weights="DEFAULT"); torch.onnx.export(..., 1x3x520x520)
//
// Usage: segmentation <deeplabv3.onnx|engine> <image> [out.jpg]

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "../common/image_io.h"

using namespace trtcpp;

namespace {

struct Rgb {
    std::uint8_t r, g, b;
};

// Standard Pascal VOC colormap (the bit-reversal palette torchvision/VOC use).
std::array<Rgb, 256> vocPalette() {
    std::array<Rgb, 256> palette{};
    for (int i = 0; i < 256; ++i) {
        int r = 0, g = 0, b = 0, c = i;
        for (int j = 0; j < 8; ++j) {
            r |= ((c >> 0) & 1) << (7 - j);
            g |= ((c >> 1) & 1) << (7 - j);
            b |= ((c >> 2) & 1) << (7 - j);
            c >>= 3;
        }
        palette[static_cast<std::size_t>(i)] = {static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b)};
    }
    return palette;
}

} // namespace

int main(int argc, char **argv) {
    const std::string root = "D:/Code/tensorrt-cpp-api";

    const std::string modelPath = root + "/models/deeplabv3_mobilenet_v3_large.onnx";
    const std::string imagePath = root + "/inputs/team.jpg";
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
            *dtypeResult == DType::kFloat32 && shape.rank() == 4 && shape[0] == 1 && shape[1] > 1 && shape[2] > 0 && shape[3] > 0;
        if (!isSegmentationLogits) {
            continue;
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
    if (!inShapeResult || inShapeResult->rank() != 4 || (*inShapeResult)[0] != 1 || (*inShapeResult)[1] != 3 ||
        (*inShapeResult)[2] <= 0 || (*inShapeResult)[3] <= 0) {
        std::fprintf(stderr, "segmentation expects input shape [1,3,H,W]\n");
        return 1;
    }
    const auto inShape = inShapeResult.value(); // [1,3,H,W]
    const int inH = static_cast<int>(inShape[2]);
    const int inW = static_cast<int>(inShape[3]);

    examples::Image img = examples::decodeImage(imagePath);
    if (img.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    Stream stream;
    auto src = examples::uploadHWC(img, stream).value();
    auto dst = Tensor::allocate(DType::kFloat32, Shape{1, 3, inH, inW}, Device::kCuda).value();
    if (auto s = preproc::letterboxToTensor(src.view(), dst.view(), examples::imagenetSpec(), stream); !s) {
        std::fprintf(stderr, "preproc: %s\n", s.message().c_str());
        return 1;
    }
    auto outputs = engine->infer({{inName, dst.view()}}, stream);
    if (!outputs) {
        std::fprintf(stderr, "infer: %s\n", outputs.status().message().c_str());
        return 1;
    }
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
    auto logits = host.as<float>().value(); // [1, C, H, W], channels-first
    const int C = static_cast<int>(host.shape()[1]);
    const int H = static_cast<int>(host.shape()[2]);
    const int W = static_cast<int>(host.shape()[3]);
    const int plane = H * W;

    // per-pixel argmax over the C class logits
    std::vector<std::uint8_t> classMap(static_cast<std::size_t>(plane));
    std::vector<int> hist(static_cast<std::size_t>(C), 0);
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

    // colorize at network resolution, nearest-neighbor resize to the original, blend 50/50
    const auto palette = vocPalette();
    examples::Image overlay = img; // copy; we blend in place
    for (int y = 0; y < img.height; ++y) {
        const int sy = y * H / img.height;
        for (int x = 0; x < img.width; ++x) {
            const int sx = x * W / img.width;
            const Rgb col = palette[classMap[static_cast<std::size_t>(sy) * W + sx]];
            auto *px = &overlay.data[(static_cast<std::size_t>(y) * img.width + x) * 3];
            px[0] = static_cast<std::uint8_t>((px[0] + col.r) / 2);
            px[1] = static_cast<std::uint8_t>((px[1] + col.g) / 2);
            px[2] = static_cast<std::uint8_t>((px[2] + col.b) / 2);
        }
    }
    examples::writeJpg(outPath, overlay);

    std::printf("segmented %dx%d image into %d classes; present classes:\n", img.width, img.height, C);
    for (int c = 0; c < C; ++c) {
        if (hist[static_cast<std::size_t>(c)] > 0) {
            std::printf("  class %2d : %5.1f%% of pixels\n", c, 100.0 * hist[static_cast<std::size_t>(c)] / plane);
        }
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
}
