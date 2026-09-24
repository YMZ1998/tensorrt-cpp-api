// OCT image segmentation example with reusable inference buffers and stage timing.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tensorrt_cpp_api/oct_segmentation.h>

namespace {

cv::Mat visualizeMask(const cv::Mat &mask, int classes) {
    cv::Mat visual;
    mask.convertTo(visual, CV_8UC1, 255.0 / static_cast<double>(classes - 1));
    return visual;
}

bool validateMask(const cv::Mat &mask, const cv::Size &expectedSize, int classes, int *hist) {
    if (mask.type() != CV_8UC1 || mask.size() != expectedSize) {
        std::fprintf(stderr, "unexpected mask shape or type\n");
        return false;
    }

    std::fill(hist, hist + classes, 0);
    for (int y = 0; y < mask.rows; ++y) {
        const auto *row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            const int classId = row[x];
            if (classId >= classes) {
                std::fprintf(stderr, "unexpected class id: %d\n", classId);
                return false;
            }
            ++hist[classId];
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    const std::string root = "D:/Code/tensorrt-cpp-api";
    const std::string modelPath = argc > 1 ? argv[1] : root + "/models/efficientnet_b1_best_model.onnx";
    const std::string imagePath = argc > 2 ? argv[2] : root + "/inputs/input.png";
    const std::string outPath = argc > 3 ? argv[3] : root + "/inputs/oct_segmentation_mask.png";
    const int iterations = 10;

    cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::fprintf(stderr, "could not read image: %s\n", imagePath.c_str());
        return 1;
    }

    auto segmenter = trtcpp::oct::OctSegmentation::Init(modelPath, root + "/models");
    if (!segmenter) {
        std::fprintf(stderr, "oct segmentation create: %s\n", segmenter.status().message().c_str());
        return 1;
    }
    cv::Mat mask;

    // The first call initializes image-size-dependent OpenCV buffers and warms TensorRT.
    // auto warmupResult = segmenter->predict(gray);
    // if (!warmupResult) {
    //    std::fprintf(stderr, "oct segmentation warmup: %s\n", warmupResult.status().message().c_str());
    //    return 1;
    //}
    segmenter->predict(gray, mask);
    const auto warmupTiming = segmenter->lastTiming();

    double preprocessTotalMs = 0.0;
    double inferenceTotalMs = 0.0;
    double postprocessTotalMs = 0.0;
    double totalMs = 0.0;

    for (int i = 0; i < iterations; ++i) {
        // auto maskResult = segmenter->predict(gray);
        // if (!maskResult) {
        //    std::fprintf(stderr, "oct segmentation predict %d: %s\n", i, maskResult.status().message().c_str());
        //    return 1;
        //}
        // mask = maskResult.value();

        segmenter->predict(gray, mask);

        // This is a shallow view of the segmenter's reusable output buffer. Consume it
        // before the next predict() call; clone only when persistent ownership is needed.
        const auto timing = segmenter->lastTiming();
        preprocessTotalMs += timing.preprocessMs;
        inferenceTotalMs += timing.inferenceMs;
        postprocessTotalMs += timing.postprocessMs;
        totalMs += timing.totalMs;
    }

    const int classes = segmenter->classes();
    int hist[4] = {};
    if (classes != 4 || !validateMask(mask, gray.size(), classes, hist)) {
        return 1;
    }

    cv::Mat visual = visualizeMask(mask, classes);
    if (!cv::imwrite(outPath, visual)) {
        std::fprintf(stderr, "could not write image: %s\n", outPath.c_str());
        return 1;
    }

    const double count = static_cast<double>(iterations);
    const double averageTotalMs = totalMs / count;
    std::printf("oct segmentation mask: %dx%d CV_8UC1\n", mask.cols, mask.rows);
    std::printf("warmup: total %.3f ms, preprocess %.3f ms, inference %.3f ms, postprocess %.3f ms\n", warmupTiming.totalMs,
                warmupTiming.preprocessMs, warmupTiming.inferenceMs, warmupTiming.postprocessMs);
    std::printf("steady state (%d runs): total %.3f ms, preprocess %.3f ms, inference %.3f ms, postprocess %.3f ms, %.2f FPS\n", iterations,
                averageTotalMs, preprocessTotalMs / count, inferenceTotalMs / count, postprocessTotalMs / count, 1000.0 / averageTotalMs);

    const double pixels = static_cast<double>(mask.rows) * mask.cols;
    for (int i = 0; i < classes; ++i) {
        if (hist[i] > 0) {
            std::printf("  class %d: %.1f%%\n", i, 100.0 * hist[i] / pixels);
        }
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
}
