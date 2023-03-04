cmake_minimum_required(VERSION 3.16)
project(hip-on-nv)

find_package(CUDA REQUIRED)
if(NOT DEFINED CUDA_PATH)
    set(CUDA_PATH /usr/local/cuda/)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_PATH}/include)

add_subdirectory(src)