#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <set>

#include <queue>
#include <unordered_set>

// 增加查询所需的数据结构
struct Neighbor {
    uint32_t id;
    float dist;
    bool operator<(const Neighbor& other) const { return dist > other.dist; } // 优先队列大顶堆
};

// 针对 A100/AVX-512 的优化预留位置
// 实际运行中可替换为更高效的距离度量库
float compute_l2(const float* a, const float* b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

struct Node {
    uint32_t id;
    std::vector<uint32_t> neighbors;
};

class CSPG_SOTA {
public:
    int dim;
    size_t num_points;
    float* data;
    std::vector<Node> graph;
    float* queries;
    uint32_t* groundtruth;
    size_t num_queries;
    int k_gt;

    // 1. 补全数据加载
    void load_queries(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        int d; in.read((char*)&d, 4);
        in.seekg(0, std::ios::end);
        num_queries = in.tellg() / (4 + d * 4);
        queries = new float[num_queries * d];
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_queries; ++i) {
            in.seekg(4, std::ios::cur);
            in.read((char*)(queries + i * d), d * 4);
        }
        std::cout << "Loaded " << num_queries << " queries." << std::endl;
    }

    void load_gt(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        in.read((char*)&k_gt, 4);
        groundtruth = new uint32_t[num_queries * k_gt];
        in.read((char*)groundtruth, num_queries * k_gt * 4);
    }

    // 2. 核心搜索逻辑：Beam Search / Greedy Search
    void search(const float* q, uint32_t K, uint32_t L, std::vector<uint32_t>& res) {
        std::priority_queue<Neighbor> pool;
        std::priority_queue<Neighbor> top_candidates; // 最小堆存最近的 L 个
        std::unordered_set<uint32_t> visited;

        uint32_t start_node = 0; // 简化处理，从0开始
        float d_start = compute_l2(q, data + start_node * dim, dim);
        pool.push({start_node, d_start});
        visited.insert(start_node);

        while (!pool.empty()) {
            Neighbor curr = pool.top(); pool.pop();
            // 如果当前节点比 top_candidates 中最远的还要远，停止该分支探索
            if (top_candidates.size() >= L && curr.dist > top_candidates.top().dist) break;

            for (uint32_t nb : graph[curr.id].neighbors) {
                if (visited.find(nb) == visited.end()) {
                    visited.insert(nb);
                    float d_nb = compute_l2(q, data + nb * dim, dim);
                    if (top_candidates.size() < L || d_nb < top_candidates.top().dist) {
                        pool.push({nb, d_nb});
                        top_candidates.push({nb, d_nb});
                        if (top_candidates.size() > L) top_candidates.pop();
                    }
                }
            }
        }
        // 最终取 K 个结果
        while(!top_candidates.empty()){
            res.push_back(top_candidates.top().id);
            top_candidates.pop();
        }
    }

    // 3. 性能测试入口
    void test_performance(uint32_t L) {
        std::cout << "Testing with L = " << L << "..." << std::endl;
        std::vector<std::vector<uint32_t>> results(num_queries);
        
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < num_queries; ++i) {
            search(queries + i * dim, 10, L, results[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_time = std::chrono::duration<double>(end - start).count();
        
        // 计算 Recall@10
        size_t hit = 0;
        for (size_t i = 0; i < num_queries; ++i) {
            std::set<uint32_t> gt_set;
            for (int j = 0; j < 10; ++j) gt_set.insert(groundtruth[i * k_gt + j]);
            for (auto res_id : results[i]) {
                if (gt_set.count(res_id)) hit++;
            }
        }
        
        std::cout << "Recall@10: " << (double)hit / (num_queries * 10) 
                  << " | QPS: " << num_queries / total_time 
                  << " | Avg Latency: " << (total_time * 1000) / num_queries << "ms" << std::endl;
    }    

    CSPG_SOTA(const std::string& path, int d) : dim(d) {
        load_fvecs(path);
        graph.resize(num_points);
        for(size_t i=0; i<num_points; ++i) graph[i].id = i;
    }

    void load_fvecs(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Error: Cannot open " << filename << std::endl;
            exit(1);
        }
        in.read((char*)&dim, 4);
        in.seekg(0, std::ios::end);
        size_t file_size = in.tellg();
        num_points = file_size / (4 + dim * 4);
        data = new float[num_points * dim];
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_points; ++i) {
            in.seekg(4, std::ios::cur);
            in.read((char*)(data + i * dim), dim * 4);
        }
        std::cout << "Loaded " << num_points << " points, dim: " << dim << std::endl;
    }

    // CSPG 核心：Crossing Pruning 剪枝
    // 逻辑：确保新加入的边不会被已有的邻居“遮挡”，且保持图的跨越式稀疏性
    void crossing_prune(uint32_t u, std::vector<uint32_t>& candidates, uint32_t R, float alpha) {
        std::vector<uint32_t> selected;
        // 1. 按距离 u 的远近排序
        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b){
            return compute_l2(data + u*dim, data + a*dim, dim) < compute_l2(data + u*dim, data + b*dim, dim);
        });

        for (uint32_t v : candidates) {
            if (v == u) continue;
            bool occluded = false;
            for (uint32_t n : selected) {
                float d_nv = compute_l2(data + n*dim, data + v*dim, dim);
                float d_uv = compute_l2(data + u*dim, data + v*dim, dim);
                
                // CSPG 核心公式：更严格的交叉遮挡判断
                if (alpha * d_nv <= d_uv) {
                    occluded = true;
                    break;
                }
            }
            if (!occluded) {
                selected.push_back(v);
                if (selected.size() >= R) break;
            }
        }
        graph[u].neighbors = selected;
    }

// 增加一个辅助搜索函数，用于建图时的候选集获取
    void search_for_build(uint32_t u, uint32_t L, std::vector<uint32_t>& candidates) {
        std::priority_queue<Neighbor> pool;
        std::priority_queue<Neighbor> top_candidates;
        std::unordered_set<uint32_t> visited;

        // 起点设为 0 或者随机
        uint32_t start_node = (u + 1) % num_points; 
        float d_start = compute_l2(data + u * dim, data + start_node * dim, dim);
        pool.push({start_node, d_start});
        visited.insert(start_node);

        while (!pool.empty()) {
            Neighbor curr = pool.top(); pool.pop();
            if (top_candidates.size() >= L && curr.dist > top_candidates.top().dist) break;

            for (uint32_t nb : graph[curr.id].neighbors) {
                if (visited.find(nb) == visited.end()) {
                    visited.insert(nb);
                    float d_nb = compute_l2(data + u * dim, data + nb * dim, dim);
                    if (top_candidates.size() < L || d_nb < top_candidates.top().dist) {
                        pool.push({nb, d_nb});
                        top_candidates.push({nb, d_nb});
                        if (top_candidates.size() > L) top_candidates.pop();
                    }
                }
            }
        }
        while(!top_candidates.empty()){
            candidates.push_back(top_candidates.top().id);
            top_candidates.pop();
        }
    }

    void build_index(uint32_t R, float alpha, uint32_t L_build) {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Building CSPG Index with Iterative Refinement..." << std::endl;
        
        // 1. 随机初始化 (Step 0)
        for (size_t i = 0; i < num_points; ++i) {
            for(uint32_t j=1; j<=R; ++j) graph[i].neighbors.push_back((i+j)%num_points);
        }

        // 2. 两轮迭代优化 (这是图索引能跑出召回率的关键)
        for (int pass = 0; pass < 2; ++pass) {
            std::cout << "Pass " << pass + 1 << " starting..." << std::endl;
            #pragma omp parallel for schedule(dynamic, 128)
            for (size_t i = 0; i < num_points; ++i) {
                std::vector<uint32_t> candidates;
                // 通过当前的图找更好的邻居候选
                search_for_build(i, L_build, candidates);
                // 执行 CSPG 核心剪枝
                crossing_prune(i, candidates, R, alpha);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Build finished in " << diff.count() << "s" << std::endl;
    }
};

int main() {
    // A100 路径直接硬连
    std::string base_path = "/home/cc/hpdic/AdaDisk/experiments/data/gist/gist_base.fvecs";
    
    // GIST1M 维度为 960
    CSPG_SOTA index(base_path, 960);
    
    // 运行构建：R=32（稀疏度）, alpha=1.2 (CSPG 常用参数)
    index.build_index(32, 1.2, 100);

    std::string root = "/home/cc/hpdic/CSPG/experiment/data/gist/";
    index.load_queries(root + "gist_query.fvecs");
    index.load_gt(root + "gist_groundtruth.ivecs");

    index.build_index(32, 1.2, 100);

    // 扫一遍不同的 L 值，拉出 Recall-QPS 曲线
    for (uint32_t L : {20, 40, 60, 80, 100, 200}) {
        index.test_performance(L);
    }    

    return 0;
}

//g++ -O3 -march=native -fopenmp cspg_clean.cpp -o cspg_clean.bin
//./cspg_clean.bin
