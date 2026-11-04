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
#include <cstdlib>

namespace diskann {

MCGIContext g_mcgi_ctx;

void RunMCGIHelloWorld() {
    std::cout << ">>> [MCGI] Module Loaded Successfully." << std::endl;
}

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

void InitMCGIContext(const std::string &lid_path, float alpha_min, float alpha_max, bool use_linear)
{
    if (lid_path.empty())
        return;

    std::cout << "[MCGI] Initializing Global Context..." << std::endl;
    std::cout << "[MCGI] Param Alpha Range: [" << alpha_min << ", " << alpha_max << "]" << std::endl;

    std::vector<float> lid_values;
    if (!LoadLIDBinary(lid_path, lid_values))
    {
        std::cerr << "[MCGI Error] Failed to load LID file. MCGI disabled." << std::endl;
        return;
    }

    g_mcgi_ctx.alpha_table.resize(lid_values.size());

    if (use_linear) {
        std::cout << "[MCGI ALGO] Using Ablation Mode: LINEAR Mapping" << std::endl;
        
        double lid_min = *std::min_element(lid_values.begin(), lid_values.end());
        double lid_max = *std::max_element(lid_values.begin(), lid_values.end());
        double lid_range = lid_max - lid_min;

        if (lid_range < 1e-6)
            lid_range = 1.0;

        std::cout << "[MCGI] LID Stats -> Min: " << lid_min << ", Max: " << lid_max << std::endl;

        for (size_t i = 0; i < lid_values.size(); ++i) {
            double ratio = (lid_values[i] - lid_min) / lid_range;
            g_mcgi_ctx.alpha_table[i] = alpha_min + (float)(ratio * (alpha_max - alpha_min));
        }
    } else {
        std::cout << "[MCGI ALGO] Using Full Mode: SIGMOID Mapping" << std::endl;
        
        double sum = 0.0;
        for (float v : lid_values)
            sum += v;
        double mean = sum / lid_values.size();

        double sq_sum = 0.0;
        for (float v : lid_values)
            sq_sum += (v - mean) * (v - mean);
        double std_dev = std::sqrt(sq_sum / lid_values.size());

        if (std_dev < 1e-6)
            std_dev = 1.0;

        std::cout << "[MCGI] LID Stats -> Mean: " << mean << ", StdDev: " << std_dev << std::endl;

        // MCGI_K / MCGI_DIR: same tuning variants as the online path in index.cpp
        const double k = std::getenv("MCGI_K") ? std::atof(std::getenv("MCGI_K")) : 1.0;
        const double dir = std::getenv("MCGI_DIR") ? std::atof(std::getenv("MCGI_DIR")) : 1.0;
        for (size_t i = 0; i < lid_values.size(); ++i) {
            double z_score = dir * (lid_values[i] - mean) / std_dev;
            double sigmoid_val = 1.0 / (1.0 + std::exp(-k * z_score));
            g_mcgi_ctx.alpha_table[i] = alpha_min + (float)(sigmoid_val * (alpha_max - alpha_min));
        }
    }

    // [MCGI] MCGI_LB_GAMMA: build search list of node i scaled by exp(gamma * z_i), normalized to mean 1,
    // so the total build effort equals the uniform list size
    if (const char *g = std::getenv("MCGI_LB_GAMMA"))
    {
        const double gamma = std::atof(g);
        double mean = 0, sq = 0;
        for (float v : lid_values)
            mean += v;
        mean /= lid_values.size();
        for (float v : lid_values)
            sq += (v - mean) * (v - mean);
        const double sd = std::max(std::sqrt(sq / lid_values.size()), 1e-6);
        g_mcgi_ctx.l_scale.resize(lid_values.size());
        double tot = 0;
        for (size_t i = 0; i < lid_values.size(); ++i)
        {
            double z = std::max(-3.0, std::min(3.0, (lid_values[i] - mean) / sd));
            g_mcgi_ctx.l_scale[i] = (float)std::exp(gamma * z);
            tot += g_mcgi_ctx.l_scale[i];
        }
        for (auto &v : g_mcgi_ctx.l_scale)
            v = (float)(v * lid_values.size() / tot);
        std::cout << "[MCGI] Build list scaled by LID, gamma=" << gamma << std::endl;
    }

    g_mcgi_ctx.enabled = true;
    std::cout << "[MCGI] Global Context Ready. Table Size: " << g_mcgi_ctx.alpha_table.size() << std::endl;

    if (lid_values.size() > 5)
    {
        std::cout << "[MCGI Debug] Sample Alphas (First 5): ";
        for (int i = 0; i < 5; ++i)
            std::cout << g_mcgi_ctx.alpha_table[i] << " ";
        std::cout << std::endl;
    }
}

// void InitMCGIContext(const std::string &lid_path, float alpha_min, float alpha_max)
// {
//     if (lid_path.empty())
//         return;

//     std::cout << "[MCGI] Initializing Global Context..." << std::endl;
//     std::cout << "[MCGI] Param Alpha Range: [" << alpha_min << ", " << alpha_max << "]" << std::endl;

//     std::vector<float> lid_values;
//     if (!LoadLIDBinary(lid_path, lid_values))
//     {
//         std::cerr << "[MCGI Error] Failed to load LID file. MCGI disabled." << std::endl;
//         return;
//     }

//     // ================= [MCGI ALGO] Sigmoid Mapping Strategy =================

//     // 1. 计算均值 (Mean)
//     double sum = 0.0;
//     for (float v : lid_values)
//         sum += v;
//     double mean = sum / lid_values.size();

//     // 2. 计算标准差 (StdDev)
//     double sq_sum = 0.0;
//     for (float v : lid_values)
//         sq_sum += (v - mean) * (v - mean);
//     double std_dev = std::sqrt(sq_sum / lid_values.size());

//     // 防止标准差过小导致除以零（极罕见情况）
//     if (std_dev < 1e-6)
//         std_dev = 1.0;

//     std::cout << "[MCGI] LID Stats -> Mean: " << mean << ", StdDev: " << std_dev << std::endl;

//     // 3. 生成 Alpha 表 (使用 Sigmoid 函数)
//     // 公式: alpha = min + (max - min) * Sigmoid( (lid - mean) / std_dev )
//     g_mcgi_ctx.alpha_table.resize(lid_values.size());

//     // 控制 Sigmoid 的陡峭程度，k 越大，在均值附近的突变越剧烈
//     // k=1.0 是标准正态分布的平滑度
//     const double k = 1.0;

//     for (size_t i = 0; i < lid_values.size(); ++i)
//     {
//         // Z-score: 将数据中心化到 0
//         double z_score = (lid_values[i] - mean) / std_dev;

//         // Sigmoid: 映射到 (0, 1)
//         double sigmoid_val = 1.0 / (1.0 + std::exp(-k * z_score));

//         // 映射到目标区间 [alpha_min, alpha_max]
//         float alpha = alpha_min + (float)(sigmoid_val * (alpha_max - alpha_min));

//         g_mcgi_ctx.alpha_table[i] = alpha;
//     }
//     // ========================================================================

//     g_mcgi_ctx.enabled = true;
//     std::cout << "[MCGI] Global Context Ready. Table Size: " << g_mcgi_ctx.alpha_table.size() << std::endl;

//     // 打印几个样本看看效果
//     if (lid_values.size() > 5)
//     {
//         std::cout << "[MCGI Debug] Sample Alphas (First 5): ";
//         for (int i = 0; i < 5; ++i)
//             std::cout << g_mcgi_ctx.alpha_table[i] << " ";
//         std::cout << std::endl;
//     }
// }

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

bool IsMCGIAdvanced()
{
    return g_mcgi_ctx.advanced;
}

void FreeMCGIContext()
{
    std::cout << "[MCGI] Cleaning up context." << std::endl;
    std::vector<float>().swap(g_mcgi_ctx.alpha_table); // 强制释放内存
    g_mcgi_ctx.enabled = false;
}

} // namespace diskann