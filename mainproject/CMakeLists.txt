include_directories(${proyecto_SOURCE_DIR}/libproject/inc)

link_directories(${proyecto_SOURCE_DIR}/libproject/src)
add_executable(Features main.cpp)
target_link_libraries(Features ${OpenCV_LIBS} ${THREAD_LIBS} libs)

add_custom_command(TARGET Features POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/mainproject/resources $<TARGET_FILE_DIR:Features>)
