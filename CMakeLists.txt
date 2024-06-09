cmake_minimum_required(VERSION 3.20)
project(pigdetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Compiler options
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set different output directories for Debug and Release
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(nlohmann_json REQUIRED)

message("=======================================")
message(" ### Find package ###")
if (OpenCV_FOUND)
    message(" OpenCV Version:   ${OpenCV_VERSION}")
else()
    message("OpenCV is not found!")
endif()
if (CUDA_FOUND)
    message(" CUDA Version:     ${CUDA_VERSION}")
else()
    message("CUDA is not found!")
endif()
if (TensorRT_FOUND)
    message(" TensorRT Version: ${TensorRT_VERSION_STRING}")
else()
    message("TensorRT is not found!")
endif()
message("=======================================")

# Define the CUDA architecture
set(CUDA_ARCH_BIN "7.5" CACHE STRING "Specify GPU architecture")
set(CUDA_ARCH_PTX "" CACHE STRING "No virtual architecture required for deployment")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN}")

### detection
set(SOURCES
    detection/src/yolov8.cpp
    detection/src/detection.cpp
)

set(HEADERS
    detection/include/common.hpp
    detection/include/yolov8.hpp
)

add_executable(detection ${SOURCES} ${HEADERS})

target_include_directories(detection PUBLIC
    ${PROJECT_SOURCE_DIR}/detection/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    )

target_link_libraries(detection PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES}
    PRIVATE nlohmann_json::nlohmann_json
)

# Debug build specific settings
set_target_properties(detection PROPERTIES DEBUG_POSTFIX "d")

### segmentation
set(SOURCES_SEG
    segmentation/src/segmentation.cpp
    segmentation/src/yolov8-seg.cpp
)

set(HEADERS_SEG
    segmentation/include/common.hpp
    segmentation/include/yolov8-seg.hpp
)

add_executable(segmentation ${SOURCES_SEG} ${HEADERS_SEG})

target_include_directories(segmentation PUBLIC
    ${PROJECT_SOURCE_DIR}/segmentation/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    )

target_link_libraries(segmentation PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES}
    PRIVATE nlohmann_json::nlohmann_json
)

# Debug build specific settings
set_target_properties(segmentation PROPERTIES DEBUG_POSTFIX "d")

### pose
set(SOURCES_POSE
    pose/src/pose.cpp
    pose/src/yolov8-pose.cpp
)

set(HEADERS_POSE
    pose/include/common.hpp
    pose/include/yolov8-pose.hpp
)

add_executable(pose ${SOURCES_POSE} ${HEADERS_POSE})

target_include_directories(pose PUBLIC
    ${PROJECT_SOURCE_DIR}/pose/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    )

target_link_libraries(pose PUBLIC
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES}
    PRIVATE nlohmann_json::nlohmann_json
)

# Debug build specific settings
set_target_properties(pose PROPERTIES DEBUG_POSTFIX "d")