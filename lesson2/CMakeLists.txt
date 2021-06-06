cmake_minimum_required(VERSION 3.2) #Setting minimum cmake version

project(NNet)  #project name main

set(CMAKE_PREFIX_PATH /home/maajid/project_files/pytorch_june6/libtorch/) # path to the pytorch c++ library is set to CMAKE_PREFIX_PATH

find_package(Torch REQUIRED) # find_package to locate all the torch packages

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(run_torch src/main.cpp
                         include/network.h) # Linking our executable run_torch with our source code (main file and the header file)

target_link_libraries(run_torch "${TORCH_LIBRARIES}") # Link the torch libs to our executbale main

set_property(TARGET run_torch PROPERTY CXX_STANDARD 14)
