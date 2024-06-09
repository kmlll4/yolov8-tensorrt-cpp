#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include "yolov8.hpp"

using namespace det;

YOLOv8::YOLOv8(const std::string& engine_file_path)
	: num_bindings(0)
	, num_inputs(0)
	, num_outputs(0)
	, engine(nullptr)
	, context(nullptr)
	, stream(nullptr)
	, gLogger(nvinfer1::ILogger::Severity::kERROR)
	, runtime(nvinfer1::createInferRuntime(gLogger))
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    initLibNvInferPlugins(&gLogger, "");

	engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
	if (engine.get() == nullptr)
	{
		std::cout << "Failed to deserialize engine." << std::endl;
		std::abort();
	};

	context = UniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	if (!context)
	{
		std::cout << "Processing error has occurred inside the create execution context." << std::endl;
		std::abort();
	}

    assert(context != nullptr);
    cudaStreamCreate(&stream);
    num_bindings = engine->getNbBindings();

    for (int i = 0; i < num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        std::string name = engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            num_inputs += 1;
            dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            input_bindings.push_back(binding);
            // set max opt shape
            context->setBindingDimensions(i, dims);
        }
        else {
            dims = context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            output_bindings.push_back(binding);
            num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
	cudaStreamDestroy(stream);

	for (auto& ptr : device_ptrs)
	{
		CHECK(cudaFree(ptr));
	}

	for (auto& ptr : host_ptrs)
	{
		CHECK(cudaFreeHost(ptr));
	}

    context.reset();
    engine.reset();
    runtime.reset();
}

void YOLOv8::make_pipe(bool warmup) {
    for (auto& bindings : input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, stream));
        device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        device_ptrs.push_back(d_ptr);
        host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, stream));
                free(h_ptr);
            }
            infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size) {
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float height = image.rows;
    float width  = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    pparam.ratio  = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image) {
    cv::Mat nchw;
    auto& in_binding = input_bindings[0];
    int width = in_binding.dims.d[3];
    int height = in_binding.dims.d[2];
    cv::Size size{width, height};
    letterbox(image, nchw, size);

    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size) {
    cv::Mat nchw;
    letterbox(image, nchw, size);
    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}

void YOLOv8::infer() {
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

void YOLOv8::postprocess(std::vector<Object>& objs) {
    objs.clear();
    int*  num_dets = static_cast<int*>(host_ptrs[0]);
    auto* boxes = static_cast<float*>(host_ptrs[1]);
    auto* scores = static_cast<float*>(host_ptrs[2]);
    int*  labels = static_cast<int*>(host_ptrs[3]);
    auto& dw = pparam.dw;
    auto& dh = pparam.dh;
    auto& width = pparam.width;
    auto& height = pparam.height;
    auto& ratio = pparam.ratio;
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

void YOLOv8::draw_objects(const cv::Mat& image,
                          cv::Mat& res,
                          const std::vector<Object>& objs,
                          const std::vector<std::string>& CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS) {
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}