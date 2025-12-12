// agent_ingest.cpp
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <cstdlib> 

namespace fs = std::filesystem;

// 1. 造数据工具
template<typename T>
void generate_data(const std::string& filename, size_t n, size_t d) {
    std::ofstream out(filename, std::ios::binary);
    int32_t n_pts = (int32_t)n;
    int32_t dim = (int32_t)d;
    out.write((char*)&n_pts, sizeof(int32_t));
    out.write((char*)&dim, sizeof(int32_t));
    std::vector<T> vec(n * d);
    for(auto& x : vec) x = (T)rand() / RAND_MAX;
    out.write((char*)vec.data(), vec.size() * sizeof(T));
    out.close();
}

int main() {
    // === 路径配置 (Ingest 侧) ===
    const std::string DIR = "./hpdic_data";
    if (!fs::exists(DIR)) fs::create_directory(DIR);

    // [关键修改] 使用 ingest_ 前缀，明确这是入库数据
    const std::string DATA_FILE = DIR + "/ingest_raw.bin";
    const std::string INDEX_PREFIX = DIR + "/ingest_index"; 
    
    // 指向 DiskANN CLI 工具
    const std::string BUILDER_BIN = "/home/cc/DiskANN/build/apps/build_disk_index";

    const size_t DIM = 128;
    const size_t NUM_POINTS = 10000; 
    const size_t NUM_THREADS = 4;

    // ---------------------------------------------------------
    // Step 1: 生成原始数据 (Ingest Raw Data)
    // ---------------------------------------------------------
    std::cout << "[Agent Ingest] Generating raw data: " << DATA_FILE << "..." << std::endl;
    generate_data<float>(DATA_FILE, NUM_POINTS, DIM);

    // ---------------------------------------------------------
    // Step 2: 构建索引 (Build Index)
    // ---------------------------------------------------------
    std::cout << "[Agent Ingest] Building DiskANN Index..." << std::endl;
    
    // 调用 build_disk_index，输入 ingest_raw.bin，输出 ingest_index_xxx
    std::string cmd = BUILDER_BIN + " "
                      "--data_type float --dist_fn l2 "
                      "--data_path " + DATA_FILE + " "
                      "--index_path_prefix " + INDEX_PREFIX + " "
                      "-R 32 -L 50 -B 0.1 -M 0.1 -T " + std::to_string(NUM_THREADS);

    std::cout << "[Command] " << cmd << std::endl;
    
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[Agent Ingest] Error: Build failed! Check builder path.\n";
        return 1;
    }

    std::cout << "[Agent Ingest] Success! Created index: " << INDEX_PREFIX << std::endl;
    return 0;
}