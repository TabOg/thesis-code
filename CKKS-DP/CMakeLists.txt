cmake_minimum_required(VERSION 3.5)
project(CKKS_DP)

add_definitions(-O3 -march=native -mtune=native -pedantic -Wall -Wextra)

find_package(SEAL 4.1 REQUIRED)
add_executable(noise noise.cpp)
target_link_libraries(noise SEAL::seal)