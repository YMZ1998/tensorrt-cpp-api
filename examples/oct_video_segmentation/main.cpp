// OCT video segmentation demo.
//
// Usage:
//   oct_video_segmentation [input_video] [output_video] [model.onnx]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <tensorrt_cpp_api/oct_segmentation.h>

namespace {

cv::Mat colorizeMask(const cv::Mat &mask) {
    static const cv::Vec3b palette[] = {
        {0, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
    };

    cv::Mat color(mask.rows, mask.cols, CV_8UC3);
    for (int y = 0; y < mask.rows; ++y) {
        const auto *src = mask.ptr<std::uint8_t>(y);
        auto *dst = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            dst[x] = palette[src[x] < 4 ? src[x] : 0];
        }
    }
    return color;
}

cv::Mat makeOverlay(const cv::Mat &frame, const cv::Mat &mask) {
    const cv::Mat colorMask = colorizeMask(mask);
    cv::Mat overlay = frame.clone();
    for (int y = 0; y < frame.rows; ++y) {
        const auto *maskRow = mask.ptr<std::uint8_t>(y);
        const auto *colorRow = colorMask.ptr<cv::Vec3b>(y);
        auto *dst = overlay.ptr<cv::Vec3b>(y);
        const auto *src = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            if (maskRow[x] == 0) {
                dst[x] = src[x];
            } else {
                dst[x] = cv::Vec3b(
                    static_cast<std::uint8_t>((static_cast<int>(src[x][0]) + colorRow[x][0]) / 2),
                    static_cast<std::uint8_t>((static_cast<int>(src[x][1]) + colorRow[x][1]) / 2),
                    static_cast<std::uint8_t>((static_cast<int>(src[x][2]) + colorRow[x][2]) / 2));
            }
        }
    }
    return overlay;
}

bool parseBoolArg(const char *value, bool fallback) {
    if (value == nullptr) {
        return fallback;
    }
    return std::string(value) != "0";
}

} // namespace

int main(int argc, char **argv) {
    const std::string root = "D:/Code/tensorrt-cpp-api";
    const std::string modelPath = argc > 3 ? argv[3] : root + "/models/efficientnet_b1_best_model.onnx";
    const std::string inputPath = argc > 1 ? argv[1] : root + "/inputs/test.mp4";
    const std::string outputPath = argc > 2 ? argv[2] : root + "/inputs/oct_segmentation_output.mp4";
    const bool showWindow = argc > 4 ? parseBoolArg(argv[4], true) : true;
    const int maxFrames = argc > 5 ? std::atoi(argv[5]) : -1;

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::fprintf(stderr, "could not open input video: %s\n", inputPath.c_str());
        return 1;
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double inputFps = capture.get(cv::CAP_PROP_FPS);
    const double outputFps = inputFps > 0.0 ? inputFps : 30.0;
    const int frameCount = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), outputFps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "could not open output video: %s\n", outputPath.c_str());
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

    std::vector<double> predictionTimes;
    double preprocessTotalMs = 0.0;
    double inferenceTotalMs = 0.0;
    double postprocessTotalMs = 0.0;
    cv::Mat frame;
    int processedFrames = 0;
    while (capture.read(frame)) {
        if (maxFrames > 0 && processedFrames >= maxFrames) {
            break;
        }
        if (frame.empty()) {
            break;
        }

        cv::Mat gray;
        if (frame.channels() == 1) {
            gray = frame;
        } else {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }

        auto maskResult = segmenter->predict(gray);
        if (!maskResult) {
            std::fprintf(stderr, "predict failed at frame %d: %s\n", processedFrames, maskResult.status().message().c_str());
            return 1;
        }

        const auto timing = segmenter->lastTiming();
        const double predictionMs = timing.totalMs;
        predictionTimes.push_back(predictionMs);
        preprocessTotalMs += timing.preprocessMs;
        inferenceTotalMs += timing.inferenceMs;
        postprocessTotalMs += timing.postprocessMs;
        cv::Mat overlay = makeOverlay(frame, maskResult.value());

        const double averageMs =
            std::accumulate(predictionTimes.begin(), predictionTimes.end(), 0.0) / static_cast<double>(predictionTimes.size());
        char text[256] = {};
        std::snprintf(
            text,
            sizeof(text),
            "frame: %d  total: %.2f ms  infer: %.2f ms  avg: %.2f ms",
            processedFrames + 1,
            predictionMs,
            timing.inferenceMs,
            averageMs);
        cv::putText(overlay, text, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        writer.write(overlay);
        ++processedFrames;

        if (showWindow) {
            cv::imshow("OCT Video Segmentation", overlay);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    }

    if (showWindow) {
        cv::destroyAllWindows();
    }

    if (processedFrames == 0) {
        std::fprintf(stderr, "input video contains no frames: %s\n", inputPath.c_str());
        return 1;
    }

    const double totalMs = std::accumulate(predictionTimes.begin(), predictionTimes.end(), 0.0);
    const double averageMs = totalMs / static_cast<double>(predictionTimes.size());
    const double averagePreprocessMs = preprocessTotalMs / static_cast<double>(processedFrames);
    const double averageInferenceMs = inferenceTotalMs / static_cast<double>(processedFrames);
    const double averagePostprocessMs = postprocessTotalMs / static_cast<double>(processedFrames);
    std::printf("processed frames: %d", processedFrames);
    if (frameCount > 0) {
        std::printf("/%d", frameCount);
    }
    std::printf("\n");
    std::printf(
        "prediction time: avg total %.3f ms, preprocess %.3f ms, inference %.3f ms, postprocess %.3f ms, %.2f FPS\n",
        averageMs,
        averagePreprocessMs,
        averageInferenceMs,
        averagePostprocessMs,
        1000.0 / averageMs);
    std::printf("wrote %s\n", outputPath.c_str());
    return 0;
}
