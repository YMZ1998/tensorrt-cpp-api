#pragma once

// Optional OCT segmentation wrapper. This header is active only when the library is
// built with TRT_CPP_API_WITH_OPENCV=ON, because the public API accepts cv::Mat.
#ifdef TRT_CPP_API_WITH_OPENCV

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "tensorrt_cpp_api/build_options.h"
#include "tensorrt_cpp_api/export.h"
#include "tensorrt_cpp_api/status.h"

namespace trtcpp::oct {

struct OctSegmentationTiming {
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double totalMs = 0.0;
};

struct OctSegmentationOptions {
    BuildOptions buildOptions;
    int expectedClasses = 4;
};

class TRT_CPP_API_EXPORT OctSegmentation {
public:
    static Result<OctSegmentation> Init(const std::string &onnxPath);
    static Result<OctSegmentation> create(const std::string &onnxPath, OctSegmentationOptions options = {});

    OctSegmentation(OctSegmentation &&) noexcept;
    OctSegmentation &operator=(OctSegmentation &&) noexcept;
    OctSegmentation(const OctSegmentation &) = delete;
    OctSegmentation &operator=(const OctSegmentation &) = delete;
    ~OctSegmentation();

    // Result<cv::Mat> predictBgr(const cv::Mat &bgr);
    bool predict(const cv::Mat &gray, cv::Mat &mask);

    OctSegmentationTiming lastTiming() const noexcept;
    cv::Size inputSize() const noexcept;
    cv::Size outputSize() const noexcept;
    int inputHeight() const noexcept;
    int inputWidth() const noexcept;
    int classes() const noexcept;

private:
    Result<cv::Mat> predict(const cv::Mat &gray);

    struct Impl;

    explicit OctSegmentation(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace trtcpp::oct

#endif // TRT_CPP_API_WITH_OPENCV
