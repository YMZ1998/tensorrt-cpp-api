#pragma once

// Optional OCT segmentation wrapper. This header is active only when the library is
// built with TRT_CPP_API_WITH_OPENCV=ON, because the public API accepts cv::Mat.
#ifdef TRT_CPP_API_WITH_OPENCV

#include <string>

#include <opencv2/core.hpp>

#include "tensorrt_cpp_api/build_options.h"
#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/status.h"

namespace trtcpp::oct {

struct OctSegmentationOptions {
    BuildOptions buildOptions;
    EngineOptions engineOptions;
    int expectedClasses = 4;
};

class OctSegmentation {
public:
    static Result<OctSegmentation> create(const std::string &onnxPath, OctSegmentationOptions options = {});

    OctSegmentation(OctSegmentation &&) noexcept;
    OctSegmentation &operator=(OctSegmentation &&) noexcept;
    OctSegmentation(const OctSegmentation &) = delete;
    OctSegmentation &operator=(const OctSegmentation &) = delete;
    ~OctSegmentation();

    Result<cv::Mat> predict(const cv::Mat &gray);

    int inputHeight() const noexcept { return inputH_; }
    int inputWidth() const noexcept { return inputW_; }
    int classes() const noexcept { return expectedClasses_; }

private:
    OctSegmentation(Engine engine, std::string inputName, std::string outputName, int inputH, int inputW, int expectedClasses);

    Engine engine_;
    Stream stream_;
    std::string inputName_;
    std::string outputName_;
    int inputH_ = 0;
    int inputW_ = 0;
    int expectedClasses_ = 4;
};

} // namespace trtcpp::oct

#endif // TRT_CPP_API_WITH_OPENCV
