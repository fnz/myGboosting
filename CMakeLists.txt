cmake_minimum_required(VERSION 3.7)
project(myGboosting)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
message(${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE CURRENT_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/*.cc
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.hxx)

message(${CURRENT_SOURCE_FILES})

set(SOURCE_FILES ${SOURCE_FILES} ${CURRENT_SOURCE_FILES})

set(CMAKE_CXX_COMPILER /Users/victor-k/Downloads/clang+llvm-6.0.0-x86_64-apple-darwin/bin/clang++)
set(CMAKE_C_COMPILER /Users/victor-k/Downloads/clang+llvm-6.0.0-x86_64-apple-darwin/bin/clang)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -Wall -Wextra -Wpedantic -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Os")

set(CMAKE_EXE_LINKER_FLAGS "-L/usr/local/lib -lprotobuf")

add_executable(myGboosting ${SOURCE_FILES})