
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(master_thesis LANGUAGES CXX CUDA)

# see if cuda is available
include(CheckLanguage)
check_language(CUDA)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# add_executable(Autoencoders Autoencoders.cpp)
add_executable(master_thesis mt.cu)
target_link_libraries(master_thesis "${CUDA_LIBRARIES}")
# set_property(TARGET master_thesis PROPERTY CXX_STANDARD 17)



