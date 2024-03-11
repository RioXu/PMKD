# CMakeList.txt: smmot-cpu 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.14)

project ("pmkd-cpu")

# 架构和指令集信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
# 设置第三方库的路径
set(EXTERNAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external")
set(PARLAYHASH_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/include/parlayhash")
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
include_directories(${PARLAYHASH_INCLUDE_DIR})

# 添加子目录cmake
# parlay
add_subdirectory("${EXTERNAL_INCLUDE_DIR}/parlaylib" EXCLUDE_FROM_ALL)
# fmtlog
add_subdirectory("${EXTERNAL_INCLUDE_DIR}/fmtlog" EXCLUDE_FROM_ALL)
# OpenSSL
find_package(OpenSSL) 

# 是否编译为merkle tree
add_compile_definitions(ENABLE_MERKLE)
add_compile_definitions(USE_PARLAY)
add_compile_definitions(USE_PARLAY_ALLOC)

# PMKD所需cpp文件
file(GLOB_RECURSE SOURCES_TREE "${CMAKE_SOURCE_DIR}/tree/*.cpp")
# PMKD
add_library (pmkd STATIC ${SOURCES_TREE})
set(PMKD_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tree")
target_include_directories(pmkd PUBLIC ${PMKD_INCLUDE_DIR})
target_link_libraries(pmkd PUBLIC parlay fmtlog-static OpenSSL::Crypto)

# tests
add_executable (test_reduce "test/test_reduce.cpp")
target_link_libraries(test_reduce PRIVATE pmkd)


add_executable (test_correctness "test/test_correctness.cpp")
target_link_libraries(test_correctness PRIVATE pmkd)

add_executable (test_save "test/test_save.cpp")
target_link_libraries(test_save PRIVATE pmkd)


add_executable (test_scale "test/test_scale.cpp")
target_link_libraries(test_scale PRIVATE pmkd)


add_executable (test_opt "test/test_opt.cpp")
target_link_libraries(test_opt PRIVATE pmkd)


add_executable (test_findbin_perf "test/test_findbin_perf.cpp")
target_link_libraries(test_findbin_perf PRIVATE pmkd)


add_executable (test_insert "test/test_insert.cpp")
target_link_libraries(test_insert PRIVATE pmkd)

add_executable (test_verify "test/test_verify.cpp")
target_link_libraries(test_verify PRIVATE pmkd)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET pmkd PROPERTY CXX_STANDARD 20)
endif()
 
# TODO: 如有需要，请添加测试并安装目标。
