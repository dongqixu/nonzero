cmake_minimum_required(VERSION 3.15)
project(nonzero)

file(GLOB SRCS *.cpp *.cu *.h)

project(nonzero LANGUAGES CXX C CUDA)

include_directories(
        /usr/local/cuda/include
        $ENV{HOME}/cub
)

set(CMAKE_CXX_STANDARD 14)

add_executable(nonzero main.cpp nms.cu nonzero.cu)
