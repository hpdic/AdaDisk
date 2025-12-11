#!/bin/bash
set -e

ROOT_DIR="$(pwd)"

echo "==> 自动修改 DiskANN 以生成共享库 libdiskann.so"

SRC_CMAKE="src/CMakeLists.txt"

# 替换 add_library(diskann ...) 为共享库
echo "==> 修改 $SRC_CMAKE"
sed -i 's/add_library(\${PROJECT_NAME} \${CPP_SOURCES})/add_library(${PROJECT_NAME} SHARED ${CPP_SOURCES})/' $SRC_CMAKE

# 重新构建
echo "==> 清理并重新构建"
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

echo "==> 编译完成"

# 查找结果
echo "==> 搜索生成的 libdiskann.so ..."
find . -name "libdiskann.so"

echo "==> 完成"
