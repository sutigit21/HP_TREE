cmake_minimum_required(VERSION 3.16)
project(HPTree VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_library(hptree INTERFACE)
target_include_directories(hptree INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(hp_tree_test src/main.cpp)
target_link_libraries(hp_tree_test PRIVATE hptree)

find_package(Threads REQUIRED)
target_link_libraries(hp_tree_test PRIVATE Threads::Threads)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(hp_tree_test PRIVATE -fsanitize=address)
endif()
