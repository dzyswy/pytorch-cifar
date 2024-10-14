project(onnx-cifar)
cmake_minimum_required(VERSION 2.8.12)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PATH_3RDPARTY ${CMAKE_SOURCE_DIR}/3rdparty)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /work/src/ncnn/build/install/)

# opencv
set(OpenCV_DIR /home/wy/src/opencv/opencv-3.4.3/out/share/OpenCV)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message("Failed to find OpenCV library")
else()
    message(STATUS "OpenCV library:")
    message(STATUS "    version: ${OpenCV_VERSION}")
	message(STATUS "    include: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "    install: ${OpenCV_INSTALL_PATH}")
	message(STATUS "    libs:    ${OpenCV_LIBS}")

    include_directories(${OpenCV_INCLUDE_DIRS})
    set(ALL_LIBS ${ALL_LIBS} ${OpenCV_LIBS})
endif()


find_package(ncnn REQUIRED)
set(ALL_LIBS ${ALL_LIBS} ncnn)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
link_directories(${PROJECT_BINARY_DIR}/libs)



add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)

