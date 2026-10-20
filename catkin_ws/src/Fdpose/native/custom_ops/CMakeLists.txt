cmake_minimum_required(VERSION 3.16)
project(torch_op)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Jetson Orin
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -std=c++17")

enable_language(CXX CUDA)

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

set(PYBIND_SRC
  src/op_test.cc
  src/op_test.cu
  torch_app/torch_op_test.cc
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
  OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(customOps ${PYBIND_SRC})
target_include_directories(customOps PRIVATE include ${TORCH_INCLUDE_DIRS})
target_link_libraries(customOps PRIVATE
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  Python3::Module
  pybind11::module
)
target_compile_definitions(customOps PRIVATE TORCH_EXTENSION_NAME=customOps)
