/**
 * @file hpdic_mcgi.h
 * @author Dongfang Zhao (dzhao@cs.washington.edu)
 * @brief Header file for Manifold-Consistent Graph Indexing (MCGI) interfaces.
 * @version 0.1
 * @date 2025-12-19
 * @copyright Copyright (c) 2025 University of Washington
 */

#pragma once
#include <vector>
#include <string>

namespace diskann
{
void RunMCGIHelloWorld();

// 1. 初始化全局 MCGI 上下文 (在 disk_utils.cpp 里调用)
void InitMCGIContext(const std::string &lid_path, float alpha_min, float alpha_max);

// 2. 获取全局 Alpha (在 index.cpp 的剪枝逻辑里调用)
// 如果没有初始化，默认返回 alpha_min (通常是 1.0 或 1.2，视你的默认策略而定)
float GetGlobalMCGIAlpha(unsigned node_id);

// 3. 清理/释放 (可选)
void FreeMCGIContext();

bool IsMCGIEnabled();
// void BuildMCGIIndex(const std::string& data_path, ...);
} // namespace diskann