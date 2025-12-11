#!/bin/bash
set -e

echo "==> 为 DiskANN 自动修补 CMake，加上 -laio"

CMAKE=src/CMakeLists.txt

# 给 shared lib 增加 aio 链接
sed -i '/add_library(${PROJECT_NAME} SHARED ${CPP_SOURCES})/a \
target_link_libraries(${PROJECT_NAME} PRIVATE aio)' $CMAKE

echo "==> 清理并重建"
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

echo "==> 完成，检查 libdiskann.so:"
find . -name "libdiskann.so"
