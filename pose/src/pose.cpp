#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8-pose.hpp"
#include "nlohmann/json.hpp"
#include <fstream>

std::vector<std::vector<unsigned int>> KPS_COLORS;
std::vector<std::vector<unsigned int>> SKELETON;
std::vector<std::vector<unsigned int>> LIMB_COLORS;

void load_class_names_and_colors(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    nlohmann::json j;
    file >> j;
    KPS_COLORS = j["kps_names"].get<std::vector<std::vector<unsigned int>>>();
    SKELETON = j["skeleton"].get<std::vector<std::vector<unsigned int>>>();
    LIMB_COLORS = j["limb_colors"].get<std::vector<std::vector<unsigned int>>>();
}


int main(int argc, char** argv)
{
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};
    const std::string config_file{"config/pose_default.json"};

    load_class_names_and_colors(config_file);

    std::vector<std::string> imagePathList;
    bool isVideo{false};

    assert(argc == 3);

    auto pose = std::make_shared<YOLOv8_pose>(engine_file_path);
    pose->make_pipe(true);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }

    cv::Mat res, image;
    cv::Size size = cv::Size{640, 640};
    int topk = 100;
    float score_thres = 0.25f;
    float iou_thres = 0.65f;

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            pose->infer();
            auto end = std::chrono::system_clock::now();
            pose->postprocess(objs, score_thres, iou_thres, topk);
            pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& path : imagePathList) {
            objs.clear();
            image = cv::imread(path);
            pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            pose->infer();
            auto end = std::chrono::system_clock::now();
            pose->postprocess(objs, score_thres, iou_thres, topk);
            pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();

    return 0;
}