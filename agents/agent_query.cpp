// agent_query.cpp
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <memory>
#include <filesystem>
#include <cstdlib>

// 只引用搜索需要的头文件
#include <pq_flash_index.h>
#include <linux_aligned_file_reader.h>

using namespace diskann;
namespace fs = std::filesystem;

// 1. 造数据函数
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
    // === 配置 ===
    const std::string DIR = "./hpdic_data";
    if (!fs::exists(DIR)) fs::create_directory(DIR);

    // [命名] 使用 query_ 前缀，与 ingest 彻底分开
    const std::string DATA_FILE = DIR + "/query_raw.bin";
    const std::string INDEX_PREFIX = DIR + "/query_index"; 
    
    // DiskANN 构建完成后会生成 _disk.index 文件，用它来判断索引是否存在
    const std::string INDEX_CHECK_FILE = INDEX_PREFIX + "_disk.index";
    
    // 指向构建工具 CLI
    const std::string BUILDER_BIN = "/home/cc/DiskANN/build/apps/build_disk_index";

    const size_t DIM = 128;
    const size_t NUM_POINTS = 10000; 
    const size_t NUM_THREADS = 4;

    // ---------------------------------------------------------
    // Step 1: 检查数据 (Data Check)
    // ---------------------------------------------------------
    if (fs::exists(DATA_FILE)) {
        std::cout << "[Agent Query] Data file exists (" << DATA_FILE << "). Skipping generation.\n";
    } else {
        std::cout << "[Agent Query] Data file missing. Generating " << NUM_POINTS << " vectors...\n";
        generate_data<float>(DATA_FILE, NUM_POINTS, DIM);
        std::cout << "[Agent Query] Data generated.\n";
    }

    // ---------------------------------------------------------
    // Step 2: 检查索引 (Index Check)
    // ---------------------------------------------------------
    if (fs::exists(INDEX_CHECK_FILE)) {
        std::cout << "[Agent Query] Index exists (" << INDEX_CHECK_FILE << "). Skipping build.\n";
    } else {
        std::cout << "[Agent Query] Index missing. Building via CLI...\n";
        
        // 调用 CLI 构建
        std::string cmd = BUILDER_BIN + " "
                          "--data_type float --dist_fn l2 "
                          "--data_path " + DATA_FILE + " "
                          "--index_path_prefix " + INDEX_PREFIX + " "
                          "-R 32 -L 50 -B 0.1 -M 0.1 -T " + std::to_string(NUM_THREADS);
        
        std::cout << "Running: " << cmd << std::endl;
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Error: Build failed! Check path: " << BUILDER_BIN << "\n";
            return 1;
        }
        std::cout << "[Agent Query] Index built successfully.\n";
    }

    // ---------------------------------------------------------
    // Step 3: 加载并搜索 (Load & Search)
    // ---------------------------------------------------------
    std::cout << "[Agent Query] Loading Index (PQFlashIndex)..." << std::endl;

    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();

    auto index = std::make_unique<PQFlashIndex<float>>(
        reader, 
        diskann::Metric::L2
    );

    int load_ret = index->load(NUM_THREADS, INDEX_PREFIX.c_str());
    if (load_ret != 0) {
        std::cerr << "Error: Load failed.\n";
        return 1;
    }
    std::cout << "Index loaded. Ready to search.\n";

    // ---------------------------------------------------------
    // Step 4: 搜索循环
    // ---------------------------------------------------------
    std::cout << "[Agent Query] Searching..." << std::endl;
    
    // 模拟搜索
    std::vector<float> query(DIM);
    for(auto& x : query) x = (float)rand() / RAND_MAX;

    std::vector<uint64_t> ids(5);
    std::vector<float> dists(5);

    index->cached_beam_search(
        query.data(), 
        5,              // K
        20,             // L_search
        ids.data(), 
        dists.data(), 
        4,              // Beam Width
        false,          // Use Reorder Data
        nullptr         // Query Stats
    );

    std::cout << "Top-1 ID: " << ids[0] << " Dist: " << dists[0] << "\n";

    return 0;
}