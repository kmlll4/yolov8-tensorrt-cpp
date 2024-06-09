# YOLOv8 TensorRT C++ Implementation

This repository provides a C++ implementation of YOLOv8 using TensorRT. It supports both object detection and segmentation tasks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Introduction

This project demonstrates how to use YOLOv8 with TensorRT for efficient inference on NVIDIA GPUs. It includes both object detection and segmentation models.

## Features

- Efficient inference with TensorRT
- Support for both object detection and segmentation
- Customizable threshold and other parameters
- Example code for inference on images and videos

## Requirements

- CUDA 11.x or later
- cuDNN 8.x or later
- TensorRT 8.x or later
- OpenCV 4.x or later
- CMake 3.18 or later
- C++17 or later

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kmlll4/yolov8-tensorrt-cpp.git
    cd yolov8-tensorrt-cpp
    ```

2. Install dependencies:
    - Make sure CUDA, cuDNN, and TensorRT are installed on your system.
    - Install OpenCV if it is not already installed:
      ```bash
      sudo apt-get update
      sudo apt-get install libopencv-dev
      ```

3. Build the project:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage

### Command-Line Arguments

- `engine_file_path`: Path to the TensorRT engine file.
- `input_path`: Path to the input image or video.

### Running the Application

To run the object detection example:
```bash
./yolov8_det engine_file_path input_path
```
To run the segmentation example:
```bash
./yolov8_seg engine_file_path input_path
```

## Example
### Object Detection Example
```bash
./yolov8_det ./models/yolov8.engine ./data/sample.jpg
```
### Segmentation Example
```bash
./yolov8_seg ./models/yolov8_seg.engine ./data/sample.jpg
```
### Configuration File
The class names and colors are stored in a JSON configuration file. Ensure that config/class_names_and_colors.json exists and contains the following structure:
```json
{
    "class_names": [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ],
    "colors": [
        [0, 114, 189], [217, 83, 25], [237, 177, 32], [126, 47, 142], [119, 172, 48], [77, 190, 238],
        [162, 20, 47], [76, 76, 76], [153, 153, 153], [255, 0, 0], [255, 128, 0], [191, 191, 0],
        [0, 255, 0], [0, 0, 255], [170, 0, 255], [85, 85, 0], [85, 170, 0], [85, 255, 0], [170, 85, 0],
        [170, 170, 0], [170, 255, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [0, 85, 128], [0, 170, 128],
        [0, 255, 128], [85, 0, 128], [85, 85, 128], [85, 170, 128], [85, 255, 128], [170, 0, 128],
        [170, 85, 128], [170, 170, 128], [170, 255, 128], [255, 0, 128], [255, 85, 128], [255, 170, 128],
        [255, 255, 128], [0, 85, 255], [0, 170, 255], [0, 255, 255], [85, 0, 255], [85, 85, 255],
        [85, 170, 255], [85, 255, 255], [170, 0, 255], [170, 85, 255], [170, 170, 255], [170, 255, 255],
        [255, 0, 255], [255, 85, 255], [255, 170, 255], [85, 0, 0], [128, 0, 0], [170, 0, 0], [212, 0, 0],
        [255, 0, 0], [0, 43, 0], [0, 85, 0], [0, 128, 0], [0, 170, 0], [0, 212, 0], [0, 255, 0], [0, 0, 43],
        [0, 0, 85], [0, 0, 128], [0, 0, 170], [0, 0, 212], [0, 0, 255], [0, 0, 0], [36, 36, 36], [73, 73, 73],
        [109, 109, 109], [146, 146, 146], [182, 182, 182], [219, 219, 219], [0, 114, 189], [80, 183, 189], [128, 128, 0]
    ]
}
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.



