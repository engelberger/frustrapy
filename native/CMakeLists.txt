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

nanobind_add_module(_core STABLE_ABI NB_STATIC ${FRUSTRAPY_NATIVE_SOURCES})
target_include_directories(_core PRIVATE src)
target_compile_definitions(_core PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)

# CPU multicore (OpenMP). Optional: when absent the core compiles serial (the
# `#ifdef _OPENMP` guards fall back to the single-threaded loops). When present it
# parallelizes the per-residue / per-contact energy reductions and the density.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "frustrapy_native: OpenMP enabled (${OpenMP_CXX_VERSION})")
else()
  message(STATUS "frustrapy_native: OpenMP not found; CPU core will be serial")
endif()

# Optional ASan/UBSan build for the N2 parity gate.
option(FRUSTRAPY_NATIVE_SANITIZE "Build with ASan/UBSan (for the N2 parity gate)" OFF)
if(FRUSTRAPY_NATIVE_SANITIZE)
  target_compile_options(_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(_core PRIVATE -fsanitize=address,undefined)
  target_compile_definitions(_core PRIVATE _GLIBCXX_ASSERTIONS)
endif()

install(TARGETS _core LIBRARY DESTINATION frustrapy_native)
