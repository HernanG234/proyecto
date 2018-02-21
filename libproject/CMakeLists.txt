include_directories(${proyecto_SOURCE_DIR}/libproject/inc)
FILE(GLOB SRCS src/*.cpp)
ADD_LIBRARY(libs ${SRCS})
target_link_libraries(libs ${OpenCV_LIBS} ${THREAD_LIBS})
