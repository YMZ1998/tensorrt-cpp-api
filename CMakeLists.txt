cmake_minimum_required(VERSION 3.22)
project(tensorrt_cpp_api VERSION 7.0.0 LANGUAGES CXX)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_INSTALL_PREFIX "D:/Code/install/oct_segmentation" CACHE PATH "Install path prefix" FORCE)

include("D:/Code/3rdparty/3rdparty.cmake")

set(PROJECT_OUTPUT_DIR "${PROJECT_BINARY_DIR}/../output" CACHE PATH "Common output directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_OUTPUT_DIR}")

# The engine core depends only on CUDA + TensorRT. The preprocessing sublibrary, OpenCV interop,
# Python bindings, tests, and examples are independent, opt-in components; all default OFF except
# the preprocessing sublibrary.

option(TRT_CPP_API_BUILD_TESTS "Build unit/integration tests" OFF)
option(TRT_CPP_API_BUILD_EXAMPLES "Build reference examples" ON)
option(TRT_CPP_API_BUILD_PREPROC "Build the fused preprocessing sublibrary" ON)
option(TRT_CPP_API_BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
option(TRT_CPP_API_WITH_OPENCV "Build the optional OpenCV interop" ON)
option(TRT_CPP_API_WITH_SPDLOG "Build the optional spdlog logger adapter" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Enable CTest at the top level so tests registered in subdirectories (tests/, python/) are
# discoverable. Called early -- before add_subdirectory(python) registers the binding test.
if(TRT_CPP_API_BUILD_TESTS)
    enable_testing()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Modern ccache hook (replaces v6's generator-specific RULE_LAUNCH_COMPILE).
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# CUDA runtime: a hard dependency of the engine core (streams, allocation, Tensor cross
# the public API). Plain C++ against the CUDA runtime API -- nvcc is only needed later for
# the preprocessing kernels (.cu), at which point CUDA is enabled as a language.
find_package(CUDAToolkit REQUIRED)

# TensorRT (apt or tarball). Linked PRIVATE: nvinfer1 types never appear in public headers
# (PImpl/forward-decl), so consumers need TRT at runtime but not at compile time. Pass
# -DTensorRT_DIR=<root> for a tarball install.
find_package(TensorRT REQUIRED)

# Windows executables do not search the TensorRT/CUDA installation directories
# automatically. Keep the reference binaries runnable from the build output and
# install tree by copying the runtime DLLs next to the executables.
function(trtcpp_windows_runtime_dlls out_var)
    if(NOT WIN32)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(_trt_root "")
    if(TENSORRT_ROOT)
        set(_trt_root "${TENSORRT_ROOT}")
    elseif(TensorRT_DIR)
        set(_trt_root "${TensorRT_DIR}")
    elseif(TensorRT_INCLUDE_DIR)
        get_filename_component(_trt_root "${TensorRT_INCLUDE_DIR}" DIRECTORY)
    endif()

    if(_trt_root)
        set(_trt_runtime_dlls "${_trt_root}/lib/nvinfer.dll" "${_trt_root}/lib/nvonnxparser.dll")
    endif()
    if(CUDNN_ROOT)
        set(_cudnn_runtime_dlls
            "${CUDNN_ROOT}/bin/cudnn64_8.dll"
            "${CUDNN_ROOT}/bin/cudnn_ops_infer64_8.dll"
            "${CUDNN_ROOT}/bin/cudnn_ops_train64_8.dll")
    endif()
    if(CUDAToolkit_BIN_DIR)
        set(_cuda_runtime_dlls
            "${CUDAToolkit_BIN_DIR}/cublasLt64_11.dll"
            "${CUDAToolkit_BIN_DIR}/cublas64_11.dll"
            "${CUDAToolkit_BIN_DIR}/cudart64_110.dll"
            "${CUDAToolkit_BIN_DIR}/cusparse64_11.dll")
    endif()

    set(_runtime_dlls
        ${_trt_runtime_dlls}
        ${_cudnn_runtime_dlls}
        ${_cuda_runtime_dlls}
    )
    set(${out_var} "${_runtime_dlls}" PARENT_SCOPE)
endfunction()

function(trtcpp_copy_windows_runtime_dlls target)
    if(NOT WIN32)
        return()
    endif()

    trtcpp_windows_runtime_dlls(_runtime_dlls)
    foreach(_dll IN LISTS _runtime_dlls)
        if(EXISTS "${_dll}")
            add_custom_command(
                TARGET ${target}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM
            )
        endif()
    endforeach()

    foreach(_opencv_lib IN LISTS OpenCV_LIBS)
        if(TARGET "${_opencv_lib}")
            get_target_property(_opencv_debug_dll "${_opencv_lib}" IMPORTED_LOCATION_DEBUG)
            get_target_property(_opencv_release_dll "${_opencv_lib}" IMPORTED_LOCATION_RELEASE)

            if(_opencv_debug_dll AND EXISTS "${_opencv_debug_dll}"
               AND _opencv_release_dll AND EXISTS "${_opencv_release_dll}")
                add_custom_command(
                    TARGET ${target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "$<$<CONFIG:Debug>:${_opencv_debug_dll}>$<$<NOT:$<CONFIG:Debug>>:${_opencv_release_dll}>"
                            "$<TARGET_FILE_DIR:${target}>"
                    VERBATIM
                )
            endif()
        endif()
    endforeach()
endfunction()

function(trtcpp_install_windows_runtime_dlls)
    if(NOT WIN32)
        return()
    endif()

    trtcpp_windows_runtime_dlls(_runtime_dlls)
    foreach(_dll IN LISTS _runtime_dlls)
        if(EXISTS "${_dll}")
            install(FILES "${_dll}" DESTINATION "${CMAKE_INSTALL_BINDIR}")
        endif()
    endforeach()

    foreach(_opencv_lib IN LISTS OpenCV_LIBS)
        if(TARGET "${_opencv_lib}")
            get_target_property(_opencv_debug_dll "${_opencv_lib}" IMPORTED_LOCATION_DEBUG)
            get_target_property(_opencv_release_dll "${_opencv_lib}" IMPORTED_LOCATION_RELEASE)

            if(_opencv_debug_dll AND EXISTS "${_opencv_debug_dll}")
                install(
                    FILES "${_opencv_debug_dll}"
                    DESTINATION "${CMAKE_INSTALL_BINDIR}"
                    CONFIGURATIONS Debug
                )
            endif()
            if(_opencv_release_dll AND EXISTS "${_opencv_release_dll}")
                install(
                    FILES "${_opencv_release_dll}"
                    DESTINATION "${CMAKE_INSTALL_BINDIR}"
                    CONFIGURATIONS Release RelWithDebInfo MinSizeRel
                )
            endif()
        endif()
    endforeach()
endfunction()

# Threads: the logger and EnginePool use std::mutex/condition_variable. Linked PRIVATE
# (PImpl, so no thread types in public headers); the package config re-resolves it.
find_package(Threads REQUIRED)

# Generate build_config.h recording the library + TensorRT versions, so public headers can
# version-gate (e.g. the legacy calibrator) without including a TensorRT header.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/tensorrt_cpp_api/build_config.h
    @ONLY
)

add_library(tensorrt_cpp_api)
add_library(tensorrt_cpp_api::tensorrt_cpp_api ALIAS tensorrt_cpp_api)
target_sources(
    tensorrt_cpp_api
    PRIVATE src/core.cpp
            src/logger.cpp
            src/cuda.cpp
            src/allocator.cpp
            src/device_tensor.cpp
            src/engine_builder.cpp
            src/engine.cpp
            src/engine_pool.cpp
            src/calibrator.cpp
            src/detail/buffers.cpp
            src/detail/calibrator_bridge.cpp
            src/detail/engine_cache.cpp
            src/detail/execution.cpp
            src/detail/sha256.cpp
            src/detail/trt_common.cpp
)
target_include_directories(tensorrt_cpp_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tensorrt_cpp_api PUBLIC CUDA::cudart PRIVATE TensorRT::TensorRT Threads::Threads)
# Build-tree include roots only. The install-tree include dir is supplied once by the
# INCLUDES DESTINATION of install(TARGETS) below (both the public headers and the generated
# build_config.h land under the same <prefix>/include tree), so no $<INSTALL_INTERFACE> here.
target_include_directories(
    tensorrt_cpp_api PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
)
target_compile_features(tensorrt_cpp_api PUBLIC cxx_std_20)
# POSITION_INDEPENDENT_CODE: the static core is routinely linked into shared objects -- the Python
# extension and downstream shared libraries such as the YOLO sibling repos' libYoloV8_TRT.so -- so
# it must be PIC. (Negligible cost; standard for an embeddable static library.)
set_target_properties(
    tensorrt_cpp_api PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON VERSION ${PROJECT_VERSION}
                                SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(NOT MSVC)
    target_compile_options(tensorrt_cpp_api PRIVATE -Wall -Wextra)
endif()

# Optional spdlog logger adapter. PUBLIC so makeSpdlogLogger() is declared and
# spdlog is linked for consumers of a spdlog-enabled install.
if(TRT_CPP_API_WITH_SPDLOG)
    find_package(spdlog CONFIG QUIET)
    if(NOT spdlog_FOUND AND SPDLOG_ROOT AND EXISTS "${SPDLOG_ROOT}/CMakeLists.txt")
        add_library(trtcpp_spdlog INTERFACE)
        add_library(spdlog::spdlog ALIAS trtcpp_spdlog)
        target_include_directories(trtcpp_spdlog INTERFACE "${SPDLOG_ROOT}/include")
        target_compile_definitions(trtcpp_spdlog INTERFACE FMT_HEADER_ONLY)
        target_link_libraries(trtcpp_spdlog INTERFACE Threads::Threads)
    endif()
    if(NOT TARGET spdlog::spdlog)
        message(FATAL_ERROR
            "Could not find or import spdlog. Set spdlog_DIR to an installed package or "
            "SPDLOG_ROOT to the spdlog source directory.")
    endif()
    target_link_libraries(tensorrt_cpp_api PUBLIC "$<BUILD_INTERFACE:spdlog::spdlog>")
    if(MSVC)
        target_compile_options(tensorrt_cpp_api PRIVATE /utf-8)
    endif()
    target_compile_definitions(tensorrt_cpp_api PUBLIC TRT_CPP_API_WITH_SPDLOG)
endif()

# Optional OpenCV interop. Requests only the `core` and `imgproc` modules (which carry
# cv::Mat and the OCT preprocessing operations) to avoid pulling opencv_dnn and its cuDNN dependency. Strictly
# opt-in; the engine core has no OpenCV in any non-gated header.
if(TRT_CPP_API_WITH_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
    target_sources(tensorrt_cpp_api PRIVATE src/opencv_interop.cpp src/oct_segmentation.cpp)
    target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS})
    target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(tensorrt_cpp_api PUBLIC TRT_CPP_API_WITH_OPENCV)
endif()

# Optional preprocessing sublibrary: one fused CUDA (.cu) kernel. Separate target --
# the engine core does NOT depend on it. Enabling it turns on the CUDA language (needs nvcc).
if(TRT_CPP_API_BUILD_PREPROC)
    # Architectures must be set BEFORE enable_language(CUDA): CMake validates the variable
    # during language init, and an unset value on hosts where nvcc can't self-detect a default
    # aborts configuration.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CUDAToolkit_VERSION VERSION_LESS "12.0")
            set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-virtual)
        endif()
    endif()
    enable_language(CUDA)
    add_library(tensorrt_cpp_api_preproc src/preproc.cu)
    add_library(tensorrt_cpp_api::preproc ALIAS tensorrt_cpp_api_preproc)
    target_link_libraries(tensorrt_cpp_api_preproc PUBLIC tensorrt_cpp_api::tensorrt_cpp_api CUDA::cudart)
    target_compile_features(tensorrt_cpp_api_preproc PUBLIC cxx_std_20)
    set_target_properties(
        tensorrt_cpp_api_preproc
        PROPERTIES CUDA_STANDARD 17
                   CUDA_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   POSITION_INDEPENDENT_CODE ON # embeddable into shared libs (Python ext, sibling .so)
                   EXPORT_NAME preproc # consumers link tensorrt_cpp_api::preproc (matches the in-tree alias)
                   VERSION ${PROJECT_VERSION}
                   SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Python bindings. Added after the core/preproc targets exist so the extension can link
# them. The subdirectory builds trtcpp._core.
if(TRT_CPP_API_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Reference examples. Normally built standalone against the installed package; this in-tree
# path is a convenience that reuses the just-built targets.
if(TRT_CPP_API_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---- install / export (find_package(tensorrt_cpp_api) consumable) ----
# Skipped under scikit-build-core (SKBUILD): a Python wheel must contain only the Python package
# and the compiled extension, not the C++ dev headers / static libs / CMake package config.
if(NOT SKBUILD)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    set(TRTCPP_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/tensorrt_cpp_api)

    set(TRTCPP_INSTALL_TARGETS tensorrt_cpp_api)
    if(TARGET tensorrt_cpp_api_preproc)
        list(APPEND TRTCPP_INSTALL_TARGETS tensorrt_cpp_api_preproc)
    endif()

    install(
        TARGETS ${TRTCPP_INSTALL_TARGETS}
        EXPORT tensorrt_cpp_apiTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    trtcpp_install_windows_runtime_dlls()

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/tensorrt_cpp_api/build_config.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorrt_cpp_api
    )
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTensorRT.cmake DESTINATION ${TRTCPP_INSTALL_CMAKEDIR})

    install(EXPORT tensorrt_cpp_apiTargets NAMESPACE tensorrt_cpp_api:: DESTINATION ${TRTCPP_INSTALL_CMAKEDIR})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorrt_cpp_api-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config.cmake INSTALL_DESTINATION ${TRTCPP_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config-version.cmake VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_cpp_api-config-version.cmake DESTINATION ${TRTCPP_INSTALL_CMAKEDIR}
    )
endif() # NOT SKBUILD

if(TRT_CPP_API_BUILD_TESTS)
    add_subdirectory(tests)
endif()
