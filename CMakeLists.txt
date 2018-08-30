cmake_minimum_required(VERSION 3.5.2)
project(minkowski)

# The version number.
set (minkowski_VERSION_MAJOR 0)
set (minkowski_VERSION_MINOR 1)

include_directories(src)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE " -pthread -std=c++11 -funroll-loops -Ofast -march=native")
set(CMAKE_CXX_FLAGS_DEBUG " -pthread -std=c++11 -g -O0 -fno-inline")

set(HEADER_FILES
    src/args.h
    src/dictionary.h
    src/minkowski.h
    src/model.h
    src/real.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/args.cc
    src/dictionary.cc
    src/minkowski.cc
    src/main.cc
    src/model.cc
    src/utils.cc
    src/vector.cc)

# Compile static library from source files
add_library(minkowski-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(minkowski-static PROPERTIES OUTPUT_NAME minkowski)

# Make executable
add_executable(minkowski-bin src/main.cc)
target_link_libraries(minkowski-bin pthread minkowski-static)
set_target_properties(minkowski-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME minkowski)

# ----------
# Unit tests
# ----------

# Add the GoogleTest repository as a subproject
add_subdirectory(googletest)

# Get all files from the /test directory that match *_test.cc
FILE(GLOB TEST_FILES test/*_test.cc)

# Add a build project 'unit-tests' built from all the test files
add_executable(unit-tests googletest/src/gtest_main.cc ${TEST_FILES})

# Link with library and add header files
target_link_libraries(unit-tests pthread gtest minkowski-static)
set_target_properties(unit-tests PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME run_tests)
