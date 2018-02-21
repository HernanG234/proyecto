CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
PROJECT(proyecto)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
include(options)

# Find OpenCV library
FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(OpenCV 3 REQUIRED) ?
include_directories(${OpenCV_INCLUDE_DIRS})

message(${OpenCV_LIBS})

# Link to threading library
set(THREAD_LIBS pthread)

add_subdirectory(mainproject)
add_subdirectory(libproject)
