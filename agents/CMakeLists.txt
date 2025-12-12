cmake_minimum_required(VERSION 3.10)
project(DiskANNAgents)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 基础依赖 (OpenMP)
find_package(OpenMP REQUIRED)
add_compile_options(-march=native -O3 -Wall -fopenmp)

# 2. DiskANN 路径
set(DISKANN_ROOT "$ENV{HOME}/DiskANN")
include_directories(${DISKANN_ROOT}/include)

# [关键 1] 强制添加库搜索路径，且放在最前
# 尝试 build/lib 和 build/src，确保覆盖 /usr/local/lib
link_directories(${DISKANN_ROOT}/build/lib)
link_directories(${DISKANN_ROOT}/build/src)

# 3. Agent Ingest (无 DiskANN 链接依赖)
add_executable(agent_ingest agent_ingest.cpp)
target_link_libraries(agent_ingest 
    OpenMP::OpenMP_CXX 
    pthread
)

# 4. Agent Query (核心修复)
add_executable(agent_query agent_query.cpp)

# [关键 2] 显式链接数学库
# 无论它抓到的是静态库还是动态库，把这些加上都能解决 cblas 报错
target_link_libraries(agent_query 
    diskann 
    aio 
    OpenMP::OpenMP_CXX 
    pthread 
    openblas   # <--- 修复 cblas_sgemm
    lapack     # <--- 修复 LAPACKE_sgesdd
)

# 5. RPATH 设置 (运行时找 .so)
set(DISKANN_LIB_DIR "${DISKANN_ROOT}/build/lib")
set_target_properties(agent_query PROPERTIES INSTALL_RPATH "${DISKANN_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)