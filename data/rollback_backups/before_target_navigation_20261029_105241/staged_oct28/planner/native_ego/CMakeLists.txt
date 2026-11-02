cmake_minimum_required(VERSION 3.18)
project(native_ego_planner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  _native_ego
  src/bindings.cpp
  src/ego_planner.cpp
)
target_include_directories(_native_ego PRIVATE include)

if(MSVC)
  target_compile_options(_native_ego PRIVATE /O2 /EHsc)
else()
  target_compile_options(_native_ego PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

set(
  NATIVE_EGO_OUTPUT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE PATH
  "Directory receiving the compiled Python extension"
)
set_target_properties(
  _native_ego
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${NATIVE_EGO_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${NATIVE_EGO_OUTPUT_DIR}"
)
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(
    _native_ego
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${configuration} "${NATIVE_EGO_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${configuration} "${NATIVE_EGO_OUTPUT_DIR}"
  )
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_executable(native_ego_core_test tests/test_core.cpp src/ego_planner.cpp)
  target_include_directories(native_ego_core_test PRIVATE include)
  if(MSVC)
    target_compile_options(native_ego_core_test PRIVATE /W4 /EHsc)
  else()
    target_compile_options(
      native_ego_core_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME native_ego_core_test COMMAND native_ego_core_test)
endif()
