cmake_minimum_required(VERSION 3.16)
project(hp_benchmark_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# tlx's CMakeLists uses an old cmake_minimum_required; force policy compat.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

FetchContent_Declare(
    tlx
    GIT_REPOSITORY https://github.com/tlx/tlx.git
    GIT_TAG        v0.6.1
)
set(TLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TLX_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tlx)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

set(HP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../cpp/include)
set(BENCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)

add_executable(hp_runner src/hp_runner.cpp)
target_include_directories(hp_runner PRIVATE
    ${HP_INCLUDE_DIR}
    ${BENCH_INCLUDE_DIR})
target_link_libraries(hp_runner PRIVATE
    nlohmann_json::nlohmann_json
    Threads::Threads)

add_executable(bplus_runner src/bplus_runner.cpp)
target_include_directories(bplus_runner PRIVATE
    ${HP_INCLUDE_DIR}
    ${BENCH_INCLUDE_DIR})
target_link_libraries(bplus_runner PRIVATE
    tlx
    nlohmann_json::nlohmann_json
    Threads::Threads)
