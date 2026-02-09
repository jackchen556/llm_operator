cmake_minimum_required(VERSION 3.18)
project(MyCutlassProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(/usr/local/cuda-12.8/include)
link_directories(/usr/local/cuda-12.8/lib64)

add_executable(main  main.cu)

# 引入头文件目录
target_include_directories(main   PRIVATE
        /data2/chenjiaken/cutlass/include
)
target_include_directories(main PRIVATE
        /data2/chenjiaken/cutlass/tools/util/include
)
target_include_directories(main  PRIVATE
        /data2/chenjiaken/cutlass/examples/common
)

set_property(TARGET main  PROPERTY CUDA_ARCHITECTURES "80")
target_link_libraries(main  cudart)