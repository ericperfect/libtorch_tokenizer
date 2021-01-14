cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Tokenizer)
message(STATUS "start running cmake...")
find_package(Boost 1.75.0 COMPONENTS system filesystem REQUIRED)
set(CMAKE_PREFIX_PATH "/home/wuyunzhao/nptextforcpp/libtorch")
set(Torch_DIR "/home/wuyunzhao/nptextforcpp/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
if (Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})

endif ()

add_executable(${PROJECT_NAME} test/main.cpp src/tokenizer.cpp src/unilib/unicode.cpp src/unilib/uninorms.cpp)

if(Boost_FOUND)

    target_link_libraries(Tokenizer ${Boost_LIBRARIES})

endif()
target_link_libraries(Tokenizer ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "src/unilib")
target_include_directories(${PROJECT_NAME} PUBLIC "src")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)