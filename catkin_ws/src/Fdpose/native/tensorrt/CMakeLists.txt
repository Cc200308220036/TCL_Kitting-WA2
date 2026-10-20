cmake_minimum_required(VERSION 3.16)
project(trt)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -std=c++17")

enable_language(CXX CUDA)

set(SRC_FILE
  src/trt_engine.cc
  plugins/common/checkMacrosPlugin.cpp
  plugins/common/plugin.cpp
  plugins/layerNormPlugin/layerNormKernel.cu
  plugins/layerNormPlugin/layerNormPlugin.cpp
)

set(INCLUDE_PATHS
  include/
  plugins/common/
  plugins/layerNormPlugin/
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include
)
find_library(CUDART_LIBRARY cudart
  HINTS /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib
)
find_library(NVINFER_LIBRARY nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib
)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT CUDART_LIBRARY OR
   NOT NVINFER_LIBRARY OR NOT NVINFER_PLUGIN_LIBRARY)
  message(FATAL_ERROR
    "TensorRT/CUDA development files were not found. Install the JetPack "
    "development packages before building pyTRTEng."
  )
endif()

list(APPEND INCLUDE_PATHS
  /usr/local/cuda/include
  ${TENSORRT_INCLUDE_DIR}
)

set(LINK_LIBS
  ${CUDART_LIBRARY}
  ${NVINFER_LIBRARY}
  ${NVINFER_PLUGIN_LIBRARY}
)

add_library(trt_lib SHARED ${SRC_FILE})
target_include_directories(trt_lib PUBLIC ${INCLUDE_PATHS})
target_link_libraries(trt_lib PUBLIC ${LINK_LIBS})

set(PYBIND_SRC
  src/trt_engine_pybind.cc
  plugins/common/checkMacrosPlugin.cpp
  plugins/common/plugin.cpp
  plugins/layerNormPlugin/layerNormKernel.cu
  plugins/layerNormPlugin/layerNormPlugin.cpp
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
  OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(pyTRTEng ${PYBIND_SRC})
target_include_directories(pyTRTEng PRIVATE ${INCLUDE_PATHS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(pyTRTEng PRIVATE
  ${LINK_LIBS}
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  Python3::Module
  pybind11::module
)
target_compile_definitions(pyTRTEng PRIVATE TORCH_EXTENSION_NAME=pyTRTEng)
