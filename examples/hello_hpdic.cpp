#include <immintrin.h>   // fix for _mm_prefetch / _MM_HINT_T*
#include <iostream>

// 不去实例化 DiskANN 类型，只包含一个轻量头或根本不包含 heavy header
int main() {
    std::cout << "compiler+linker smoke test OK\n";
    return 0;
}
