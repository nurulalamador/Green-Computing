cmake_minimum_required(VERSION 3.18)

project(LibTorchBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(libtorch_benchmark main.cpp)

target_link_libraries(libtorch_benchmark
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS}
)

target_include_directories(libtorch_benchmark PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

set_property(TARGET libtorch_benchmark PROPERTY CXX_STANDARD 17)

if (MSVC)
    target_compile_options(libtorch_benchmark PRIVATE /EHsc)
    target_link_libraries(libtorch_benchmark Psapi)

    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

    add_custom_command(TARGET libtorch_benchmark
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:libtorch_benchmark>
    )
endif()