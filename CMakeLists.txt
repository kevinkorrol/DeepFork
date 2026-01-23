cmake_minimum_required(VERSION 3.18)
project(deepfork_cpp)

set(CMAKE_CXX_STANDARD 17)
#set(INCLUDE_DIRECTORIES ./)
include_directories(extern/pybind11/include)
#
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#
#execute_process(
#        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
#        OUTPUT_VARIABLE TORCH_PREFIX
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#        RESULT_VARIABLE TORCH_SCRIPT_RESULT
#)
#
#if(NOT TORCH_SCRIPT_RESULT EQUAL 0)
#    message(FATAL_ERROR "Could not find Torch. Ensure it is installed.")
#endif()
#
#list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
#if(TORCH_CUDA_ARCH_LIST OR CUDA_FOUND)
#    message(STATUS "CUDA capability detected via PyTorch.")
#    add_definitions(-DUSE_CUDA)
#else()
#    message(STATUS "No CUDA detected. Building for CPU only.")
#endif()
#
add_subdirectory(extern/pybind11)
#
file(GLOB SOURCES "cpp/*.cpp")
pybind11_add_module(deepfork_cpp ${SOURCES})

add_library(test_lib STATIC cpp/chess_utils.cpp)
target_include_directories (test_lib PUBLIC extern/pybind11/include)

target_link_libraries(deepfork_cpp PRIVATE ${TORCH_LIBRARIES} pybind11::module)
target_include_directories (deepfork_cpp PUBLIC extern/pybind11/include)
#target_include_directories(deepfork_cpp PRIVATE extern/chess-library/include)

if(MSVC)
    target_compile_options(deepfork_cpp PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(deepfork_cpp PRIVATE -O3 -march=native)
endif()
