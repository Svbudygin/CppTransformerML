cmake_minimum_required(VERSION 3.10)

project(Framework1)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 20)

# the following options prevent compiler-optimization issues that are unwanted in an edu process
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

include_directories(${PROJECT_SOURCE_DIR}/modules)

file(GLOB SOURCES "modules/*.cpp")

add_executable(Framework1 ${SOURCES})

set_target_properties(Framework1 PROPERTIES WIN32_EXECUTABLE FALSE)
