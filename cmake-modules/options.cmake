
###############################################
# declare program options with default values #
###############################################

######################
# Set compiler flags #
######################

## Enable most warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

## Disable annoying Eigen warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

## Enable C++11 support
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-all-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpeel-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftracer")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
set(CMAKE_AR "gcc-ar")
set(CMAKE_RANLIB "gcc-ranlib")
