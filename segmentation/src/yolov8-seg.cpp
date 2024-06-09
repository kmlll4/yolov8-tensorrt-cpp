#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "yolov8-seg.hpp"

using namespace seg;

YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path)
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
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);

    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    delete[] trtModelStream;
    context = engine->createExecutionContext();

    assert(context != nullptr);
    cudaStreamCreate(&stream);
    num_bindings = engine->getNbBindings();

    for (int i = 0; i < num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        std::string name  = engine->getBindingName(i);
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

YOLOv8_seg::~YOLOv8_seg()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaStreamDestroy(stream);
    for (auto& ptr : device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_seg::make_pipe(bool warmup)
{
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

void YOLOv8_seg::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
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

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    pparam.ratio  = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;
}

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image) {
    cv::Mat  nchw;
    auto& in_binding = input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    cv::Size size{width, height};
    letterbox(image, nchw, size);

    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image, cv::Size& size) {
    cv::Mat nchw;
    letterbox(image, nchw, size);
    context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, stream));
}

void YOLOv8_seg::infer() {
    context->enqueueV2(device_ptrs.data(), stream, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);
}

void YOLOv8_seg::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int seg_channels, int seg_h, int seg_w) {
    objs.clear();
    auto input_h = input_bindings[0].dims.d[2];
    auto input_w = input_bindings[0].dims.d[3];
    auto num_anchors = output_bindings[0].dims.d[1];
    auto num_channels = output_bindings[0].dims.d[2];

    auto& dw = pparam.dw;
    auto& dh = pparam.dh;
    auto& width = pparam.width;
    auto& height = pparam.height;
    auto& ratio = pparam.ratio;

    auto* output = static_cast<float*>(host_ptrs[0]);
    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(host_ptrs[1]));

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr   = output + i * num_channels;
        float  score = *(ptr + 4);
        if (score > score_thres) {
            float x0 = *ptr++ - dw;
            float y0 = *ptr++ - dh;
            float x1 = *ptr++ - dw;
            float y1 = *ptr++ - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);

            int label = *(++ptr);
            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, ++ptr);
            mask_confs.push_back(mask_conf);
            labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
        }
    }

#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif
    cv::Mat masks;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    if (masks.empty()) {
        // masks is empty
    }

    else {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat   = matmulRes.reshape(indices.size(), {seg_h, seg_w});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}

void YOLOv8_seg::draw_objects(const cv::Mat& image,
                              cv::Mat& res,
                              const std::vector<Object>& objs,
                              const std::vector<std::string>& CLASS_NAMES,
                              const std::vector<std::vector<unsigned int>>& COLORS,
                              const std::vector<std::vector<unsigned int>>& MASK_COLORS) {
    res = image.clone();
    cv::Mat mask = image.clone();
    for (auto& obj : objs) {
        int idx = obj.label;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color = cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);
        mask(obj.rect).setTo(mask_color, obj.boxMask);

        int baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}