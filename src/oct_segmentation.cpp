#include "tensorrt_cpp_api/oct_segmentation.h"

#ifdef TRT_CPP_API_WITH_OPENCV

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorrt_cpp_api/device_tensor.h"
#include "tensorrt_cpp_api/dtype.h"
#include "tensorrt_cpp_api/engine_builder.h"
#include "tensorrt_cpp_api/shape.h"
#include "tensorrt_cpp_api/tensor.h"

namespace trtcpp::oct {
namespace {

float maskedGrayAt(const cv::Mat &gray, int x, int y) {
    x = std::clamp(x, 0, gray.cols - 1);
    y = std::clamp(y, 0, gray.rows - 1);

    const float cx = (static_cast<float>(gray.cols) - 1.0f) * 0.5f;
    const float cy = (static_cast<float>(gray.rows) - 1.0f) * 0.5f;
    const float radius = static_cast<float>(std::min(gray.cols, gray.rows)) * 0.5f;
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    if (dx * dx + dy * dy > radius * radius) {
        return 0.0f;
    }

    return static_cast<float>(gray.ptr<std::uint8_t>(y)[x]);
}

std::vector<float> preprocessOctMinusOneToOne(const cv::Mat &gray, int outH, int outW) {
    std::vector<float> chw(static_cast<std::size_t>(outH) * outW);
    const float scaleY = static_cast<float>(gray.rows) / static_cast<float>(outH);
    const float scaleX = static_cast<float>(gray.cols) / static_cast<float>(outW);

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

            const float v00 = maskedGrayAt(gray, x0, y0);
            const float v01 = maskedGrayAt(gray, x1, y0);
            const float v10 = maskedGrayAt(gray, x0, y1);
            const float v11 = maskedGrayAt(gray, x1, y1);
            const float top = v00 + (v01 - v00) * wx;
            const float bottom = v10 + (v11 - v10) * wx;
            const float pixel = top + (bottom - top) * wy;

            chw[static_cast<std::size_t>(y) * outW + x] = pixel / 127.5f - 1.0f;
        }
    }

    return chw;
}

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

OctSegmentation::OctSegmentation(Engine engine, std::string inputName, std::string outputName, int inputH, int inputW, int expectedClasses)
    : engine_(std::move(engine)), inputName_(std::move(inputName)), outputName_(std::move(outputName)), inputH_(inputH), inputW_(inputW),
      expectedClasses_(expectedClasses) {}

OctSegmentation::OctSegmentation(OctSegmentation &&) noexcept = default;
OctSegmentation &OctSegmentation::operator=(OctSegmentation &&) noexcept = default;
OctSegmentation::~OctSegmentation() = default;

Result<OctSegmentation> OctSegmentation::create(const std::string &onnxPath, OctSegmentationOptions options) {
    if (options.expectedClasses <= 0) {
        return Status{StatusCode::kInvalidArgument, "expectedClasses must be greater than zero"};
    }

    TRTCPP_TRY(auto engine, EngineBuilder{}.buildAndLoad(onnxPath, options.buildOptions, options.engineOptions));

    const auto inputNames = engine.inputNames();
    if (inputNames.size() != 1) {
        return Status{StatusCode::kInvalidArgument, "OCT segmentation expects exactly one input"};
    }

    const std::string inputName = inputNames.front();
    TRTCPP_TRY(auto inputShape, engine.tensorShape(inputName));
    if (inputShape.rank() != 4 || inputShape[0] != 1 || inputShape[1] != 1 || inputShape[2] <= 0 || inputShape[3] <= 0) {
        return Status{StatusCode::kShapeMismatch, "OCT segmentation input must have shape [1,1,H,W]"};
    }

    TRTCPP_TRY(auto outputName, findOutputName(engine, options.expectedClasses));

    return OctSegmentation{std::move(engine), inputName, outputName, static_cast<int>(inputShape[2]), static_cast<int>(inputShape[3]),
                           options.expectedClasses};
}

Result<cv::Mat> OctSegmentation::predict(const cv::Mat &gray) {
    if (gray.empty()) {
        return Status{StatusCode::kInvalidArgument, "input cv::Mat is empty"};
    }
    if (gray.type() != CV_8UC1) {
        return Status{StatusCode::kInvalidArgument, "OCT segmentation input must be CV_8UC1"};
    }

    const cv::Mat input = gray.isContinuous() ? gray : gray.clone();
    std::vector<float> inputHost = preprocessOctMinusOneToOne(input, inputH_, inputW_);

    TRTCPP_TRY(auto deviceInput, Tensor::allocate(DType::kFloat32, Shape{1, 1, inputH_, inputW_}, Device::kCuda));
    TensorView hostInput{inputHost.data(), DType::kFloat32, Shape{1, 1, inputH_, inputW_}, Device::kHost};
    if (auto status = deviceInput.copyFrom(hostInput, stream_); !status) {
        return status;
    }

    TRTCPP_TRY(auto outputs, engine_.infer({{inputName_, deviceInput.view()}}, stream_));
    auto outIt = outputs.find(outputName_);
    if (outIt == outputs.end()) {
        return Status{StatusCode::kInternal, "OCT segmentation output tensor was not returned"};
    }

    TRTCPP_TRY(auto hostOutput, outIt->second.toHost(stream_));
    const Shape &outShape = hostOutput.shape();
    if (hostOutput.dtype() != DType::kFloat32 || outShape.rank() != 4 || outShape[0] != 1 || outShape[1] != expectedClasses_ ||
        outShape[2] <= 0 || outShape[3] <= 0) {
        return Status{StatusCode::kShapeMismatch, "OCT segmentation output must be float32 with shape [1,4,H,W]"};
    }

    TRTCPP_TRY(auto logits, hostOutput.as<float>());
    const int classes = static_cast<int>(outShape[1]);
    const int outH = static_cast<int>(outShape[2]);
    const int outW = static_cast<int>(outShape[3]);
    const int plane = outH * outW;

    std::vector<std::uint8_t> classMap(static_cast<std::size_t>(plane));
    for (int p = 0; p < plane; ++p) {
        int best = 0;
        float bestVal = logits[static_cast<std::size_t>(p)];
        for (int c = 1; c < classes; ++c) {
            const float value = logits[static_cast<std::size_t>(c) * plane + p];
            if (value > bestVal) {
                bestVal = value;
                best = c;
            }
        }
        classMap[static_cast<std::size_t>(p)] = static_cast<std::uint8_t>(best);
    }

    cv::Mat mask(gray.rows, gray.cols, CV_8UC1);
    for (int y = 0; y < mask.rows; ++y) {
        const int sy = y * outH / mask.rows;
        std::uint8_t *row = mask.ptr<std::uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            const int sx = x * outW / mask.cols;
            row[x] = classMap[static_cast<std::size_t>(sy) * outW + sx];
        }
    }

    return mask;
}

} // namespace trtcpp::oct

#endif // TRT_CPP_API_WITH_OPENCV
