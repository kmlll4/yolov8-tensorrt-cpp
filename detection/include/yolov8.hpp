#pragma onece

#include "NvInferPlugin.h"
#include "common.hpp"

namespace det {

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void infer();
    void postprocess(std::vector<Object>& objs);
    static void draw_objects(const cv::Mat& image,
                            cv::Mat& res,
                            const std::vector<Object>& objs,
                            const std::vector<std::string>& CLASS_NAMES,
                            const std::vector<std::vector<unsigned int>>& COLORS);
private:
	UniquePtr<nvinfer1::IRuntime> runtime;
	UniquePtr<nvinfer1::ICudaEngine> engine;
	UniquePtr<nvinfer1::IExecutionContext> context;
	cudaStream_t stream;
	Logger gLogger;
	int num_bindings;
	int num_inputs;
	int num_outputs;
	std::vector<Binding> input_bindings;
	std::vector<Binding> output_bindings;
	std::vector<void*> host_ptrs;
	std::vector<void*> device_ptrs;
	PreParam pparam;
};

} // namespace det