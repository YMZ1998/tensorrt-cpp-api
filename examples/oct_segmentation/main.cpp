// Smoke test for the OpenCV-facing OCT segmentation wrapper.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tensorrt_cpp_api/oct_segmentation.h>

namespace {

cv::Mat visualizeMask(const cv::Mat &mask, int classes) {
    cv::Mat visual(mask.rows, mask.cols, CV_8UC1);
    for (int y = 0; y < mask.rows; ++y) {
        const auto *src = mask.ptr<std::uint8_t>(y);
        auto *dst = visual.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            dst[x] = static_cast<std::uint8_t>(src[x] * 255 / (classes - 1));
        }
    }
    return visual;
}

} // namespace

int main() {
    const std::string root = "D:/Code/tensorrt-cpp-api";
    const std::string modelPath = root + "/models/efficientnet_b1_best_model.onnx";
    const std::string imagePath = root + "/inputs/input.png";
    const std::string outPath = root + "/inputs/oct_segmentation_mask.png";

    cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    trtcpp::oct::OctSegmentationOptions options;
    options.buildOptions.precision = trtcpp::Precision::kFp16;
    options.buildOptions.engineCacheDir = root + "/models";

    auto segmenter = trtcpp::oct::OctSegmentation::create(modelPath, std::move(options));
    if (!segmenter) {
        std::fprintf(stderr, "oct segmentation create: %s\n", segmenter.status().message().c_str());
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    auto maskResult = segmenter->predict(gray);
    const auto end = std::chrono::steady_clock::now();
    if (!maskResult) {
        std::fprintf(stderr, "oct segmentation predict: %s\n", maskResult.status().message().c_str());
        return 1;
    }
    const double predictMs = std::chrono::duration<double, std::milli>(end - start).count();

    cv::Mat mask = std::move(maskResult.value());
    if (mask.type() != CV_8UC1 || mask.rows != gray.rows || mask.cols != gray.cols) {
        std::fprintf(stderr, "unexpected mask shape or type\n");
        return 1;
    }

    int hist[4] = {};
    for (int y = 0; y < mask.rows; ++y) {
        const auto *row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] >= 4) {
                std::fprintf(stderr, "unexpected class id: %u\n", static_cast<unsigned>(row[x]));
                return 1;
            }
            ++hist[row[x]];
        }
    }

    if (!cv::imwrite(outPath, visualizeMask(mask, segmenter->classes()))) {
        std::fprintf(stderr, "could not write image: %s\n", outPath.c_str());
        return 1;
    }

    std::printf("oct segmentation mask: %dx%d CV_8UC1\n", mask.cols, mask.rows);
    std::printf("prediction time: %.3f ms\n", predictMs);
    for (int i = 0; i < 4; ++i) {
        if (hist[i] > 0) {
            std::printf("  class %d: %.1f%%\n", i, 100.0 * hist[i] / (mask.rows * mask.cols));
        }
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
}
