cmake_minimum_required(VERSION 3.15...3.30)
project(frustrapy_native LANGUAGES CXX)

# Memory-safe modern C++ (C++ Core Guidelines): C++23 where the compiler supports it,
# libstdc++ assertions on so out-of-bounds span/vector access traps in debug builds.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)   # fall back to C++20 (std::span) on older toolchains
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Locate Python and nanobind (nanobind ships its CMake config via `python -m nanobind`).
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(FRUSTRAPY_NATIVE_SOURCES src/bindings.cpp src/core.cpp)

# Optional CUDA backend (N3). OFF by default so a no-GPU machine (this container, no
# nvcc) still builds the CPU core. When ON and nvcc is found, the .cu sources compile in
# and has_cuda() returns True; when nvcc is absent the build proceeds CPU-only.
option(FRUSTRAPY_NATIVE_CUDA "Compile the optional CUDA kernels (requires nvcc)" OFF)
if(FRUSTRAPY_NATIVE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND FRUSTRAPY_NATIVE_SOURCES src/cuda/kernels.cu)
    add_compile_definitions(FRUSTRAPY_NATIVE_CUDA)
    message(STATUS "frustrapy_native: CUDA enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "FRUSTRAPY_NATIVE_CUDA=ON but no CUDA compiler found; building CPU-only")
  endif()
endif()

# Optional Metal backend (G1.4, Apple GPUs). OFF by default so the no-GPU container
# (no `xcrun metal`, no metal-cpp) still builds the CPU core. When ON it requires a Mac
# (APPLE), the metal-cpp single-header binding (point FRUSTRAPY_METAL_CPP_DIR at its
# include root), and the `xcrun metal`/`metallib` compiler. The MSL kernels are
# precompiled to kernels.metallib (custom command) and installed next to the extension;
# dispatch.cpp loads that metallib at runtime. See native/docs/METAL_BUILD.md.
set(FRUSTRAPY_NATIVE_METAL_ENABLED OFF)
option(FRUSTRAPY_NATIVE_METAL "Compile the optional Metal kernels (Apple, requires metal-cpp + xcrun metal)" OFF)
if(FRUSTRAPY_NATIVE_METAL)
  if(NOT APPLE)
    message(WARNING "FRUSTRAPY_NATIVE_METAL=ON but not building on Apple; building without Metal")
  else()
    set(FRUSTRAPY_METAL_CPP_DIR "" CACHE PATH "Path to the metal-cpp headers (the dir containing Metal/Metal.hpp)")
    if(NOT FRUSTRAPY_METAL_CPP_DIR OR NOT EXISTS "${FRUSTRAPY_METAL_CPP_DIR}/Metal/Metal.hpp")
      message(FATAL_ERROR
        "FRUSTRAPY_NATIVE_METAL=ON requires metal-cpp. Set FRUSTRAPY_METAL_CPP_DIR to the "
        "metal-cpp include root (the directory containing Metal/Metal.hpp), e.g. "
        "-C cmake.define.FRUSTRAPY_METAL_CPP_DIR=/path/to/metal-cpp. See native/docs/METAL_BUILD.md.")
    endif()
    find_program(XCRUN_EXECUTABLE xcrun)
    if(NOT XCRUN_EXECUTABLE)
      message(FATAL_ERROR "FRUSTRAPY_NATIVE_METAL=ON but `xcrun` was not found (install the Xcode command line tools).")
    endif()
    list(APPEND FRUSTRAPY_NATIVE_SOURCES src/metal/dispatch.cpp)
    set(FRUSTRAPY_NATIVE_METAL_ENABLED ON)
    message(STATUS "frustrapy_native: Metal enabled (metal-cpp at ${FRUSTRAPY_METAL_CPP_DIR})")
  endif()
endif()

nanobind_add_module(_core STABLE_ABI NB_STATIC ${FRUSTRAPY_NATIVE_SOURCES})
target_include_directories(_core PRIVATE src)
target_compile_definitions(_core PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)

if(FRUSTRAPY_NATIVE_METAL_ENABLED)
  target_include_directories(_core PRIVATE "${FRUSTRAPY_METAL_CPP_DIR}")
  target_compile_definitions(_core PRIVATE FRUSTRAPY_NATIVE_METAL)
  target_link_libraries(_core PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")

  # Precompile the MSL kernels to a .metallib and install it alongside the extension so
  # dispatch.cpp can load it at runtime (it also accepts a FRUSTRAPY_METALLIB override).
  set(FRUSTRAPY_METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
  set(FRUSTRAPY_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")
  add_custom_command(
    OUTPUT "${FRUSTRAPY_METALLIB}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal" -o "${FRUSTRAPY_METAL_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${FRUSTRAPY_METAL_AIR}" -o "${FRUSTRAPY_METALLIB}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal"
    COMMENT "frustrapy_native: compiling MSL kernels -> kernels.metallib"
    VERBATIM)
  add_custom_target(frustrapy_metallib ALL DEPENDS "${FRUSTRAPY_METALLIB}")
  add_dependencies(_core frustrapy_metallib)
  install(FILES "${FRUSTRAPY_METALLIB}" DESTINATION frustrapy_native)
endif()

# Optional ASan/UBSan build for the N2 parity gate.
option(FRUSTRAPY_NATIVE_SANITIZE "Build with ASan/UBSan (for the N2 parity gate)" OFF)
if(FRUSTRAPY_NATIVE_SANITIZE)
  target_compile_options(_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(_core PRIVATE -fsanitize=address,undefined)
  target_compile_definitions(_core PRIVATE _GLIBCXX_ASSERTIONS)
endif()

install(TARGETS _core LIBRARY DESTINATION frustrapy_native)
