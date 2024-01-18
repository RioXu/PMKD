# CMakeList.txt: smmot-cpu 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.14)

project ("pmkd-cpu")

# 设置第三方库的路径
set(EXTERNAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external")
# 设置parlay并行线程数
set(PARLAY_NUM_THREADS 16)

# 项目内除第三方库以外的所有cpp文件
#file(GLOB_RECURSE SOURCES_THIS "${CMAKE_SOURCE_DIR}/*.cpp")
# 第三方库的所有cpp文件
#file(GLOB_RECURSE SOURCES_EXT "${EXTERNAL_INCLUDE_DIR}/*.cpp")
# 去除SOURCES_THIS中包含的SOURCES_EXT
#list(REMOVE_ITEM SOURCES_THIS ${SOURCES_EXT})

# 全局include
include_directories(${CMAKE_SOURCE_DIR})

# 添加子目录cmake
# parlay
add_subdirectory("${EXTERNAL_INCLUDE_DIR}/parlaylib" EXCLUDE_FROM_ALL)
# fmtlog
add_subdirectory("${EXTERNAL_INCLUDE_DIR}/fmtlog" EXCLUDE_FROM_ALL)

# PMKD所需cpp文件
file(GLOB_RECURSE SOURCES_TREE "${CMAKE_SOURCE_DIR}/tree/*.cpp")
# PMKD
add_library (pmkd STATIC ${SOURCES_TREE})
set(PMKD_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tree")
target_include_directories(pmkd PUBLIC ${PMKD_INCLUDE_DIR})
target_link_libraries(pmkd PUBLIC parlay)

# main
add_executable (pmkd-cpu "${CMAKE_SOURCE_DIR}/parkd-cpu.cpp")
target_link_libraries(pmkd-cpu PRIVATE pmkd)

# tests
add_executable (test_reduce "test/test_reduce.cpp")
target_link_libraries(test_reduce PRIVATE pmkd fmtlog-static)


add_executable (test_correctness "test/test_correctness.cpp")
target_link_libraries(test_correctness PRIVATE pmkd fmtlog-static)

add_executable (test_save "test/test_save.cpp")
target_link_libraries(test_save PRIVATE pmkd fmtlog-static)


add_executable (test_scale "test/test_scale.cpp")
target_link_libraries(test_scale PRIVATE pmkd fmtlog-static)


add_executable (test_opt "test/test_opt.cpp")
target_link_libraries(test_opt PRIVATE pmkd fmtlog-static)


add_executable (test_query_perf "test/test_query_perf.cpp")
target_link_libraries(test_query_perf PRIVATE pmkd fmtlog-static)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET pmkd PROPERTY CXX_STANDARD 20)
  set_property(TARGET pmkd-cpu PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
