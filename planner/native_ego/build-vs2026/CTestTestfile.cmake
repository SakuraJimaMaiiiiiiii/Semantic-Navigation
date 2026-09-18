# CMake generated Testfile for 
# Source directory: E:/files/Semantic Navigation/planner/native_ego
# Build directory: E:/files/Semantic Navigation/planner/native_ego/build-vs2026
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(native_ego_core_test "E:/files/Semantic Navigation/planner/native_ego/build-vs2026/Debug/native_ego_core_test.exe")
  set_tests_properties(native_ego_core_test PROPERTIES  _BACKTRACE_TRIPLES "E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;56;add_test;E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(native_ego_core_test "E:/files/Semantic Navigation/planner/native_ego/build-vs2026/Release/native_ego_core_test.exe")
  set_tests_properties(native_ego_core_test PROPERTIES  _BACKTRACE_TRIPLES "E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;56;add_test;E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(native_ego_core_test "E:/files/Semantic Navigation/planner/native_ego/build-vs2026/MinSizeRel/native_ego_core_test.exe")
  set_tests_properties(native_ego_core_test PROPERTIES  _BACKTRACE_TRIPLES "E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;56;add_test;E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(native_ego_core_test "E:/files/Semantic Navigation/planner/native_ego/build-vs2026/RelWithDebInfo/native_ego_core_test.exe")
  set_tests_properties(native_ego_core_test PROPERTIES  _BACKTRACE_TRIPLES "E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;56;add_test;E:/files/Semantic Navigation/planner/native_ego/CMakeLists.txt;0;")
else()
  add_test(native_ego_core_test NOT_AVAILABLE)
endif()
