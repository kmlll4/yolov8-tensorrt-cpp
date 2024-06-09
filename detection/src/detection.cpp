#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include "nlohmann/json.hpp"
#include <fstream>

std::vector<std::string> CLASS_NAMES;
std::vector<std::vector<unsigned int>> COLORS;

void load_class_names_and_colors(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    nlohmann::json j;
    file >> j;
    CLASS_NAMES = j["class_names"].get<std::vector<std::string>>();
    COLORS = j["colors"].get<std::vector<std::vector<unsigned int>>>();
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};
    const std::string config_file{"config/dtc_default.json"};

    load_class_names_and_colors(config_file);

    std::vector<std::string> imagePathList;
    bool isVideo{false};

    assert(argc == 3);

    auto detector = std::make_shared<det::YOLOv8>(engine_file_path);
    detector->make_pipe(true);

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
    std::vector<det::Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            detector->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            detector->infer();
            auto end = std::chrono::system_clock::now();
            detector->postprocess(objs);
            detector->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
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
            detector->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            detector->infer();
            auto end = std::chrono::system_clock::now();
            detector->postprocess(objs);
            detector->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();

    return 0;
}