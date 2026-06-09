cmake_minimum_required(VERSION 3.15...3.30)
project(frustramol_tmol LANGUAGES CXX)

# The torch-free CPU energy kernels are plain C++17 (std::array/std::vector + OpenMP).
# CPU-only by default, exactly like the AWSEM native core's default build. Two optional GPU
# lanes share this same kernel math and are OFF by default so they never touch the
# in-container CPU build: the CUDA lane (M6, FRUSTRAMOL_TMOL_CUDA) and the Metal lane (M7,
# FRUSTRAMOL_TMOL_METAL, Apple GPUs). Both mirror native/.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CPU multicore (OpenMP). Optional: when absent the drivers compile serial (the
# `#ifdef _OPENMP` guards fall back to the single-threaded loop), exactly like native/.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "frustramol_tmol: OpenMP enabled (${OpenMP_CXX_VERSION})")
else()
  message(STATUS "frustramol_tmol: OpenMP not found; CPU kernels will be serial")
endif()

# Optional CUDA forward path (M6). OFF by default so the no-GPU container (no nvcc) still
# builds the CPU harness and module, exactly like the AWSEM native core (native/). When ON
# and nvcc is found, the .cu kernels (src/cuda/kernels.cu) compile into the Python module
# and into a standalone CUDA parity harness, and FRUSTRAMOL_TMOL_CUDA is defined so the
# binding exposes has_cuda()=True and routes use_cuda=True to the GPU path. The device math
# mirrors the parity-gated CPU kernel headers term-for-term; the maintainer builds and
# validates parity on a CUDA host. When nvcc is absent the build proceeds CPU-only. The
# Metal lane (M7) declares its own FRUSTRAMOL_TMOL_METAL option further down; both knobs
# match the native/ install-command shape (the M9 packaging tiers), and requesting a GPU
# path that was not built is a hard error, never a silent CPU fallback.
set(FRUSTRAMOL_TMOL_CUDA_ENABLED OFF)
option(FRUSTRAMOL_TMOL_CUDA "Compile the optional CUDA forward kernels (requires nvcc)" OFF)
if(FRUSTRAMOL_TMOL_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(FRUSTRAMOL_TMOL_CUDA_ENABLED ON)
    message(STATUS "frustramol_tmol: CUDA enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING "FRUSTRAMOL_TMOL_CUDA=ON but no CUDA compiler found; building CPU-only")
  endif()
endif()

# ---------------------------------------------------------------------------
# Standalone CPU parity harness (no torch, no nanobind). Always built: this is the
# in-container-validated build path. It loads a tmol-webgpu fixture and checks the C++
# CPU drivers reproduce tmol's per-residue-pair energies to the tmol gate tolerance.
# ---------------------------------------------------------------------------
option(FRUSTRAMOL_TMOL_HARNESS "Build the standalone CPU parity harness" ON)
if(FRUSTRAMOL_TMOL_HARNESS)
  add_executable(tmol_parity_harness tests/parity_main.cpp)
  target_include_directories(tmol_parity_harness PRIVATE include tests)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tmol_parity_harness PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_definitions(tmol_parity_harness
                             PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

# ---------------------------------------------------------------------------
# Standalone CUDA parity harness (maintainer, GPU host). Built only when CUDA is enabled:
# runs the CPU driver AND the CUDA path on the same fixture and asserts they agree (and
# both agree with the tmol oracle) to the gate tolerance. Mirrors the CPU harness above and
# the AWSEM native CUDA validation lane. No GPU in the container, so this never builds here.
# ---------------------------------------------------------------------------
if(FRUSTRAMOL_TMOL_CUDA_ENABLED)
  add_executable(tmol_cuda_parity_harness tests/cuda_parity_main.cu src/cuda/kernels.cu)
  target_include_directories(tmol_cuda_parity_harness PRIVATE include tests)
  target_compile_definitions(tmol_cuda_parity_harness PRIVATE FRUSTRAMOL_TMOL_CUDA)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tmol_cuda_parity_harness PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# ---------------------------------------------------------------------------
# Optional Metal backend (M7, Apple GPUs). OFF by default so the no-GPU container (no
# `xcrun metal`, no metal-cpp) builds the CPU kernels unchanged. When ON it requires a Mac
# (APPLE), the metal-cpp single-header binding (point FRUSTRAMOL_TMOL_METAL_CPP_DIR at its
# include root), and the `xcrun metal`/`metallib` compiler. The MSL kernels are precompiled
# to kernels.metallib (custom command) and installed next to the extension; dispatch.cpp
# loads that metallib at runtime. Mirrors the AWSEM Metal lane (../native/CMakeLists.txt).
# See docs/tmol/M7_NATIVE_METAL.md.
# ---------------------------------------------------------------------------
set(FRUSTRAMOL_TMOL_METAL_ENABLED OFF)
option(FRUSTRAMOL_TMOL_METAL "Compile the optional Metal kernels (Apple, requires metal-cpp + xcrun metal)" OFF)
if(FRUSTRAMOL_TMOL_METAL)
  if(NOT APPLE)
    message(WARNING "FRUSTRAMOL_TMOL_METAL=ON but not building on Apple; building without Metal")
  else()
    set(FRUSTRAMOL_TMOL_METAL_CPP_DIR "" CACHE PATH "Path to the metal-cpp headers (the dir containing Metal/Metal.hpp)")
    if(NOT FRUSTRAMOL_TMOL_METAL_CPP_DIR OR NOT EXISTS "${FRUSTRAMOL_TMOL_METAL_CPP_DIR}/Metal/Metal.hpp")
      message(FATAL_ERROR
        "FRUSTRAMOL_TMOL_METAL=ON requires metal-cpp. Set FRUSTRAMOL_TMOL_METAL_CPP_DIR to the "
        "metal-cpp include root (the directory containing Metal/Metal.hpp), e.g. "
        "-C cmake.define.FRUSTRAMOL_TMOL_METAL_CPP_DIR=/path/to/metal-cpp. See docs/tmol/M7_NATIVE_METAL.md.")
    endif()
    find_program(XCRUN_EXECUTABLE xcrun)
    if(NOT XCRUN_EXECUTABLE)
      message(FATAL_ERROR "FRUSTRAMOL_TMOL_METAL=ON but `xcrun` was not found (install the Xcode command line tools).")
    endif()
    set(FRUSTRAMOL_TMOL_METAL_ENABLED ON)
    message(STATUS "frustramol_tmol: Metal enabled (metal-cpp at ${FRUSTRAMOL_TMOL_METAL_CPP_DIR})")
  endif()
endif()

# ---------------------------------------------------------------------------
# Optional nanobind Python module (the maintainer host / pip build, mirroring native/).
# Built when nanobind + a Python development module are found (scikit-build-core puts
# them on CMAKE_PREFIX_PATH during `pip install ./native_tmol`). A bare standalone
# `cmake -S native_tmol` without nanobind just builds the harness above and skips this.
# ---------------------------------------------------------------------------
find_package(Python 3.9 QUIET COMPONENTS Interpreter Development.Module)
set(_nb_found OFF)
if(Python_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nb_rc ERROR_QUIET)
  if(_nb_rc EQUAL 0)
    find_package(nanobind CONFIG QUIET)
    if(nanobind_FOUND)
      set(_nb_found ON)
    endif()
  endif()
endif()

if(_nb_found)
  message(STATUS "frustramol_tmol: nanobind found; building the Python module")
  set(_kernels_sources src/bindings.cpp)
  if(FRUSTRAMOL_TMOL_CUDA_ENABLED)
    list(APPEND _kernels_sources src/cuda/kernels.cu)
  endif()
  if(FRUSTRAMOL_TMOL_METAL_ENABLED)
    list(APPEND _kernels_sources src/metal/dispatch.cpp)
  endif()
  nanobind_add_module(_kernels STABLE_ABI NB_STATIC ${_kernels_sources})
  target_include_directories(_kernels PRIVATE include)
  target_compile_definitions(_kernels PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
  if(FRUSTRAMOL_TMOL_CUDA_ENABLED)
    target_compile_definitions(_kernels PRIVATE FRUSTRAMOL_TMOL_CUDA)
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(FRUSTRAMOL_TMOL_METAL_ENABLED)
    target_include_directories(_kernels PRIVATE "${FRUSTRAMOL_TMOL_METAL_CPP_DIR}")
    target_compile_definitions(_kernels PRIVATE FRUSTRAMOL_TMOL_METAL)
    target_link_libraries(_kernels PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")

    # Precompile the MSL kernels to a .metallib and install it alongside the extension so
    # dispatch.cpp can load it at runtime (it also accepts a FRUSTRAMOL_TMOL_METALLIB env
    # override).
    set(FRUSTRAMOL_TMOL_METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
    set(FRUSTRAMOL_TMOL_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")
    add_custom_command(
      OUTPUT "${FRUSTRAMOL_TMOL_METALLIB}"
      COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c
              "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal" -o "${FRUSTRAMOL_TMOL_METAL_AIR}"
      COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${FRUSTRAMOL_TMOL_METAL_AIR}" -o "${FRUSTRAMOL_TMOL_METALLIB}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels.metal"
      COMMENT "frustramol_tmol: compiling MSL kernels -> kernels.metallib"
      VERBATIM)
    add_custom_target(frustramol_tmol_metallib ALL DEPENDS "${FRUSTRAMOL_TMOL_METALLIB}")
    add_dependencies(_kernels frustramol_tmol_metallib)
    install(FILES "${FRUSTRAMOL_TMOL_METALLIB}" DESTINATION frustramol_tmol)
  endif()

  install(TARGETS _kernels LIBRARY DESTINATION frustramol_tmol)
else()
  message(STATUS "frustramol_tmol: nanobind not found; Python module skipped "
                 "(standalone harness only)")
endif()

install(DIRECTORY include/ DESTINATION include)
