cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(dcgan)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG master
)
FetchContent_MakeAvailable(argparse)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)

target_link_libraries(dcgan "${TORCH_LIBRARIES}" argparse)

set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)