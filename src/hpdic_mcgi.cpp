/**
 * @file hpdic_mcgi.cpp
 * @author Dongfang Zhao (dzhao@cs.washington.edu)
 * @brief Implementation of Manifold-Consistent Graph Indexing (MCGI) logic.
 * @version 0.1
 * @date 2025-12-19
 * * @copyright Copyright (c) 2025 University of Washington
 * */

#include "hpdic_mcgi.h"
#include <iostream>

namespace diskann {

void RunMCGIHelloWorld() {
    std::cout << ">>> [MCGI] Module Loaded Successfully." << std::endl;
    std::cout << ">>> [MCGI] Ready to implement LID estimation logic." << std::endl;
}

// --- 内部私有类和静态实例 ---
struct MCGIContext
{
    bool enabled = false;
    std::vector<float> alpha_table;
    float default_alpha = 1.2f; // 默认值
};

// 全局静态实例 (Hidden from outside)
static MCGIContext g_mcgi_ctx;

// --- 实现全局接口 ---

void InitMCGIContext(const std::string &lid_path, float alpha_min, float alpha_max)
{
    if (lid_path.empty())
        return;

    std::cout << "[MCGI] Initializing Global Context..." << std::endl;

    // ... (此处填入你之前写的读取文件和计算 Sigmoid 的逻辑) ...
    // ... 把算好的值填入 g_mcgi_ctx.alpha_table ...

    g_mcgi_ctx.enabled = true;
    std::cout << "[MCGI] Global Context Ready." << std::endl;
}

float GetGlobalMCGIAlpha(unsigned node_id)
{
    // 极速路径：如果没有启用，直接返回默认值
    if (!g_mcgi_ctx.enabled)
        return 1.2f; 

    // 边界检查
    if (node_id >= g_mcgi_ctx.alpha_table.size())
        return 1.2f;

    return g_mcgi_ctx.alpha_table[node_id];
}

void FreeMCGIContext()
{
    g_mcgi_ctx.alpha_table.clear();
    g_mcgi_ctx.enabled = false;
}

}