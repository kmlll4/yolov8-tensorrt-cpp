#pragma once

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace pose;

class YOLOv8_pose {
public:
    explicit YOLOv8_pose(const std::string& engine_file_path);

    ~YOLOv8_pose();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    static void draw_objects(const cv::Mat& image,
                             cv::Mat& res,
                             const std::vector<Object>& objs,
                             const std::vector<std::vector<unsigned int>>& SKELETON,
                             const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                             const std::vector<std::vector<unsigned int>>& LIMB_COLORS);

    int num_bindings;
    int num_inputs  = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;
    PreParam pparam;

private:
    nvinfer1::ICudaEngine* engine  = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream  = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};