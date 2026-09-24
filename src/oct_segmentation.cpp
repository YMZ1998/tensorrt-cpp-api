#include "tensorrt_cpp_api/oct_segmentation.h"

#ifdef TRT_CPP_API_WITH_OPENCV

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "tensorrt_cpp_api/cuda.h"
#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/engine.h"
#include "tensorrt_cpp_api/engine_builder.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp::oct {
namespace {

Result<std::string> findOutputName(Engine &engine, int expectedClasses) {
    const auto outputNames = engine.outputNames();
    for (const std::string &name : outputNames) {
        auto shapeResult = engine.tensorShape(name);
        auto dtypeResult = engine.tensorDType(name);
        if (!shapeResult || !dtypeResult) {
            continue;
        }

        const Shape &shape = shapeResult.value();
        if (*dtypeResult == DType::kFloat32 && shape.rank() == 4 && shape[0] == 1 && shape[1] == expectedClasses && shape[2] > 0 &&
            shape[3] > 0) {
            return name;
        }
    }
    return Status{StatusCode::kShapeMismatch, "OCT segmentation output must be float32 with shape [1,4,H,W]"};
}

} // namespace

struct OctSegmentation::Impl {
    Impl(Engine engine, std::string inputName, std::string outputName, int inputH, int inputW, int outputH, int outputW,
         int expectedClasses)
        : engine(std::move(engine)), inputName(std::move(inputName)), outputName(std::move(outputName)), inputH(inputH), inputW(inputW),
          outputH(outputH), outputW(outputW), expectedClasses(expectedClasses) {}

    Status initializeBuffers() {
        const Shape inputShape{1, 1, inputH, inputW};
        const Shape outputShape{1, expectedClasses, outputH, outputW};

        TRTCPP_TRY(auto newInputHost, Tensor::allocate(DType::kFloat32, inputShape, Device::kHost));
        TRTCPP_TRY(auto newInputDevice, Tensor::allocate(DType::kFloat32, inputShape, Device::kCuda));
        TRTCPP_TRY(auto newOutputDevice, Tensor::allocate(DType::kFloat32, outputShape, Device::kCuda));
        TRTCPP_TRY(auto newOutputHost, Tensor::allocate(DType::kFloat32, outputShape, Device::kHost));
        inputHost = std::move(newInputHost);
        inputDevice = std::move(newInputDevice);
        outputDevice = std::move(newOutputDevice);
        outputHost = std::move(newOutputHost);

        floatInput = cv::Mat(inputH, inputW, CV_32FC1, inputHost.data());
        classMap.create(outputH, outputW, CV_8UC1);
        inputs.clear();
        outputs.clear();
        inputs.emplace(inputName, inputDevice.view());
        outputs.emplace(outputName, outputDevice.view());
        return {};
    }

    Status preparePreprocessBuffers(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            return Status{StatusCode::kInvalidArgument, "input cv::Mat dimensions must be positive"};
        }
        if (!roiMask.empty() && roiMask.rows == rows && roiMask.cols == cols) {
            return {};
        }

        roiMask.create(rows, cols, CV_8UC1);
        roiMask.setTo(cv::Scalar::all(0));
        const cv::Point center((cols - 1) / 2, (rows - 1) / 2);
        const int radius = std::min(rows, cols) / 2;
        cv::circle(roiMask, center, radius, cv::Scalar::all(255), cv::FILLED, cv::LINE_8);

        maskedGray.create(rows, cols, CV_8UC1);
        resizedGray.create(inputH, inputW, CV_8UC1);
        bgrGray.create(rows, cols, CV_8UC1);
        mask.create(rows, cols, CV_8UC1);
        return {};
    }

    Result<cv::Mat> predictGray(const cv::Mat &gray) {
        const auto totalStart = std::chrono::steady_clock::now();
        timing = {};

        if (gray.empty()) {
            return Status{StatusCode::kInvalidArgument, "input cv::Mat is empty"};
        }
        if (gray.type() != CV_8UC1) {
            return Status{StatusCode::kInvalidArgument, "OCT segmentation input must be CV_8UC1"};
        }
        if (auto status = preparePreprocessBuffers(gray.rows, gray.cols); !status) {
            return status;
        }

        const auto preprocessStart = std::chrono::steady_clock::now();
        cv::bitwise_and(gray, roiMask, maskedGray);
        cv::resize(maskedGray, resizedGray, resizedGray.size(), 0.0, 0.0, cv::INTER_LINEAR);
        resizedGray.convertTo(floatInput, CV_32F, 1.0 / 127.5, -1.0);
        timing.preprocessMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preprocessStart).count();

        const auto inferenceStart = std::chrono::steady_clock::now();
        if (auto status = inputDevice.copyFrom(inputHost.view(), stream); !status) {
            return status;
        }
        if (auto status = engine.enqueue(inputs, outputs, stream); !status) {
            return status;
        }
        if (auto status = outputHost.copyFrom(outputDevice.view(), stream); !status) {
            return status;
        }
        if (auto status = stream.synchronize(); !status) {
            return status;
        }
        timing.inferenceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inferenceStart).count();

        const auto postprocessStart = std::chrono::steady_clock::now();
        TRTCPP_TRY(auto logits, outputHost.as<float>());
        const int plane = outputH * outputW;
        const float *logitData = logits.data();
        for (int p = 0; p < plane; ++p) {
            const float value0 = logitData[p];
            const float value1 = logitData[plane + p];
            const float value2 = logitData[2 * plane + p];
            const float value3 = logitData[3 * plane + p];
            int best = 0;
            float bestValue = value0;
            if (value1 > bestValue) {
                bestValue = value1;
                best = 1;
            }
            if (value2 > bestValue) {
                bestValue = value2;
                best = 2;
            }
            if (value3 > bestValue) {
                best = 3;
            }
            classMap.ptr<std::uint8_t>()[p] = static_cast<std::uint8_t>(best);
        }

        cv::resize(classMap, mask, mask.size(), 0.0, 0.0, cv::INTER_NEAREST);
        timing.postprocessMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - postprocessStart).count();
        timing.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - totalStart).count();
        return mask;
    }

    Result<cv::Mat> predictBgr(const cv::Mat &bgr) {
        if (bgr.empty()) {
            return Status{StatusCode::kInvalidArgument, "input cv::Mat is empty"};
        }
        if (bgr.type() != CV_8UC3) {
            return Status{StatusCode::kInvalidArgument, "OCT segmentation BGR input must be CV_8UC3"};
        }
        if (auto status = preparePreprocessBuffers(bgr.rows, bgr.cols); !status) {
            return status;
        }
        cv::cvtColor(bgr, bgrGray, cv::COLOR_BGR2GRAY);
        return predictGray(bgrGray);
    }

    Engine engine;
    Stream stream;
    std::string inputName;
    std::string outputName;
    int inputH = 0;
    int inputW = 0;
    int outputH = 0;
    int outputW = 0;
    int expectedClasses = 4;
    Tensor inputHost;
    Tensor inputDevice;
    Tensor outputDevice;
    Tensor outputHost;
    std::unordered_map<std::string, TensorView> inputs;
    std::unordered_map<std::string, TensorView> outputs;
    cv::Mat roiMask;
    cv::Mat maskedGray;
    cv::Mat resizedGray;
    cv::Mat floatInput;
    cv::Mat classMap;
    cv::Mat mask;
    cv::Mat bgrGray;
    OctSegmentationTiming timing;
};

OctSegmentation::OctSegmentation(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

OctSegmentation::OctSegmentation(OctSegmentation &&) noexcept = default;
OctSegmentation &OctSegmentation::operator=(OctSegmentation &&) noexcept = default;
OctSegmentation::~OctSegmentation() = default;

Result<OctSegmentation> OctSegmentation::Init(const std::string &onnxPath, const std::string &engineCacheDir) {
    OctSegmentationOptions options;
    options.buildOptions.precision = Precision::kFp16;
    options.buildOptions.engineCacheDir = engineCacheDir;
    return create(onnxPath, std::move(options));
}

Result<OctSegmentation> OctSegmentation::create(const std::string &onnxPath, OctSegmentationOptions options) {
    if (options.expectedClasses != 4) {
        return Status{StatusCode::kInvalidArgument, "OCT segmentation expects exactly four output classes"};
    }

    TRTCPP_TRY(auto engine, EngineBuilder{}.buildAndLoad(onnxPath, options.buildOptions, EngineOptions{}));

    const auto inputNames = engine.inputNames();
    if (inputNames.size() != 1) {
        return Status{StatusCode::kInvalidArgument, "OCT segmentation expects exactly one input"};
    }

    const std::string inputName = inputNames.front();
    TRTCPP_TRY(auto inputShape, engine.tensorShape(inputName));
    if (inputShape.isDynamic() || inputShape.rank() != 4 || inputShape[0] != 1 || inputShape[1] != 1 || inputShape[2] <= 0 ||
        inputShape[3] <= 0) {
        return Status{StatusCode::kShapeMismatch, "OCT segmentation input must have shape [1,1,H,W]"};
    }

    TRTCPP_TRY(auto outputName, findOutputName(engine, options.expectedClasses));
    TRTCPP_TRY(auto outputShape, engine.tensorShape(outputName));
    if (outputShape.isDynamic() || outputShape.rank() != 4 || outputShape[0] != 1 || outputShape[1] != 4 || outputShape[2] <= 0 ||
        outputShape[3] <= 0) {
        return Status{StatusCode::kShapeMismatch, "OCT segmentation output must have shape [1,4,H,W]"};
    }

    std::unique_ptr<Impl> impl{new (std::nothrow) Impl{std::move(engine), inputName, outputName, static_cast<int>(inputShape[2]),
                                                       static_cast<int>(inputShape[3]), static_cast<int>(outputShape[2]),
                                                       static_cast<int>(outputShape[3]), options.expectedClasses}};
    if (!impl) {
        return Status{StatusCode::kInternal, "failed to allocate OCT segmentation state"};
    }
    if (auto status = impl->initializeBuffers(); !status) {
        return status;
    }
    return OctSegmentation{std::move(impl)};
}

Result<cv::Mat> OctSegmentation::predict(const cv::Mat &gray) { return impl_->predictGray(gray); }

//Result<cv::Mat> OctSegmentation::predictBgr(const cv::Mat &bgr) { return impl_->predictBgr(bgr); }

bool OctSegmentation::predict(const cv::Mat &gray, cv::Mat &mask) {
    auto maskResult = impl_->predictGray(gray);
    if (!maskResult) {
        std::fprintf(stderr, "oct segmentation predict  %s\n", maskResult.status().message().c_str());
        return false;
    }
    mask = maskResult.value();
    return true;
}

OctSegmentationTiming OctSegmentation::lastTiming() const noexcept { return impl_ ? impl_->timing : OctSegmentationTiming{}; }

cv::Size OctSegmentation::inputSize() const noexcept { return impl_ ? cv::Size(impl_->inputW, impl_->inputH) : cv::Size{}; }

cv::Size OctSegmentation::outputSize() const noexcept { return impl_ ? cv::Size(impl_->outputW, impl_->outputH) : cv::Size{}; }

int OctSegmentation::inputHeight() const noexcept { return impl_ ? impl_->inputH : 0; }

int OctSegmentation::inputWidth() const noexcept { return impl_ ? impl_->inputW : 0; }

int OctSegmentation::classes() const noexcept { return impl_ ? impl_->expectedClasses : 0; }

} // namespace trtcpp::oct

#endif // TRT_CPP_API_WITH_OPENCV
