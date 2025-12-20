/**
 * @file hpdic_mcgi.cpp
 * @brief Implementation of Manifold-Consistent Graph Indexing (MCGI) logic.
 * @copyright Copyright (c) 2025 Dongfang Zhao (dzhao@uw.edu)
 */

#include "hpdic_mcgi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace diskann {

void RunMCGIHelloWorld() {
    std::cout << ">>> [MCGI] Module Loaded Successfully." << std::endl;
}

// --- 内部私有类和静态实例 ---
struct MCGIContext
{
    bool enabled = false;
    std::vector<float> alpha_table;
};

// 全局静态实例 (Hidden from outside)
static MCGIContext g_mcgi_ctx;

// --- 辅助函数：读取 DiskANN 二进制文件 ---
// 格式: int32(num_points) | int32(dim) | data...
bool LoadLIDBinary(const std::string& path, std::vector<float>& lid_values) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[MCGI Error] Cannot open LID file: " << path << std::endl;
        return false;
    }

    int32_t npts_i32, ndim_i32;
    in.read((char*)&npts_i32, sizeof(int32_t));
    in.read((char*)&ndim_i32, sizeof(int32_t));

    if (npts_i32 <= 0) return false;

    size_t num_points = (size_t)npts_i32;
    lid_values.resize(num_points);
    
    // 读取 float 数组
    in.read((char*)lid_values.data(), num_points * sizeof(float));
    
    std::cout << "[MCGI] Loaded " << num_points << " LID values." << std::endl;
    return true;
}

// --- 实现全局接口 ---

void InitMCGIContext(const std::string &lid_path, float alpha_min, float alpha_max)
{
    if (lid_path.empty()) return;

    std::cout << "[MCGI] Initializing Global Context..." << std::endl;
    std::cout << "[MCGI] Param Alpha Range: [" << alpha_min << ", " << alpha_max << "]" << std::endl;

    std::vector<float> lid_values;
    if (!LoadLIDBinary(lid_path, lid_values)) {
        std::cerr << "[MCGI Error] Failed to load LID file. MCGI disabled." << std::endl;
        return;
    }

    // 1. 找到 LID 的最大最小值用于归一化
    float min_lid = 1e30f, max_lid = -1e30f;
    for (float v : lid_values) {
        if (v < min_lid) min_lid = v;
        if (v > max_lid) max_lid = v;
    }

    // 避免除以零
    if (std::abs(max_lid - min_lid) < 1e-6) max_lid = min_lid + 1.0f;

    // 2. 计算 Alpha 表
    // 逻辑：LID 越大（越难），Alpha 应该越大（增加搜索广度）
    // 公式：alpha = alpha_min + (normalized_lid) * (alpha_max - alpha_min)
    g_mcgi_ctx.alpha_table.resize(lid_values.size());
    
    for (size_t i = 0; i < lid_values.size(); ++i) {
        float normalized = (lid_values[i] - min_lid) / (max_lid - min_lid);
        
        // 简单的线性映射 (你可以换成 Sigmoid 或其他非线性映射)
        float alpha = alpha_min + normalized * (alpha_max - alpha_min);
        
        g_mcgi_ctx.alpha_table[i] = alpha;
    }

    g_mcgi_ctx.enabled = true;
    std::cout << "[MCGI] Global Context Ready. Table Size: " << g_mcgi_ctx.alpha_table.size() << std::endl;
}

float GetGlobalMCGIAlpha(unsigned node_id)
{
    if (!g_mcgi_ctx.enabled) return 1.2f; // 默认 fallback

    if (node_id >= g_mcgi_ctx.alpha_table.size()) return 1.2f;

    return g_mcgi_ctx.alpha_table[node_id];
}

// [FIX] 这就是你缺失的那个函数，导致编译报错的原因
bool IsMCGIEnabled()
{
    return g_mcgi_ctx.enabled;
}

void FreeMCGIContext()
{
    std::cout << "[MCGI] Cleaning up context." << std::endl;
    std::vector<float>().swap(g_mcgi_ctx.alpha_table); // 强制释放内存
    g_mcgi_ctx.enabled = false;
}

} // namespace diskann