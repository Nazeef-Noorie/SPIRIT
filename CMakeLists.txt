cmake_minimum_required(VERSION 3.30.0)
project(SPIRIT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug/Release/RelWithDebInfo)" FORCE)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
endif()

include_directories(include)

file(GLOB SRC_FILES src/*.cpp)

add_executable(spirit ${SRC_FILES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(spirit PUBLIC OpenMP::OpenMP_CXX)
endif()

