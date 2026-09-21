```bash
#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "========================================"
echo " TensorRT ONNX Model Downloader"
echo "========================================"
echo
echo "[INFO] Model directory:"
echo "       $MODEL_DIR"
echo

download_model()
{
    local name="$1"
    local url="$2"

    echo "Downloading: $name"

    if [ -f "$name" ]; then
        echo "  [OK] Already exists."
        echo
        return
    fi

    curl -fL \
        --retry 5 \
        --retry-delay 2 \
        --connect-timeout 10 \
        --progress-bar \
        -o "$name" \
        "$url"

    if [ -f "$name" ]; then
        echo "  [OK] Downloaded."
    else
        echo "  [ERROR] Download failed."
        exit 1
    fi

    echo
}

# 1. Classification
download_model \
    "mobilenetv2-7.onnx" \
    "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"

# 2. Detection
download_model \
    "yolov8n.onnx" \
    "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"

# 3. Segmentation
download_model \
    "deeplabv3_mobilenet_v3_large.onnx" \
    "https://huggingface.co/ENOT-AutoDL/coco-benchmark/resolve/main/deeplabv3_mobilenet_v3_large/deeplabv3_mobilenet_v3_large.onnx?download=true"

echo "========================================"
echo " Result"
echo "========================================"
echo

ls -lh *.onnx

echo
echo "[OK] All models are in:"
echo "     $MODEL_DIR"
echo
```
