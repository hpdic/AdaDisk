#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <set>

struct RetMeta {
    uint32_t id;
    float dist;
    bool expanded;
    bool operator<(const RetMeta& o) const { return dist < o.dist; }
};

struct Neighbor {
    uint32_t id;
    float dist;
    bool operator<(const Neighbor& other) const { return dist < other.dist; }
};

class CSPG {
public:
    int dim;
    size_t num_points = 0, num_queries = 0;
    float *data = nullptr, *queries = nullptr;
    uint32_t *groundtruth = nullptr;
    int k_gt = 0;
    std::vector<std::vector<uint32_t>> graph;

    CSPG(const std::string& base_p, const std::string& query_p, const std::string& gt_p, int d) : dim(d) {
        load_data(base_p, query_p, gt_p);
    }

    float compute_l2(const float* a, const float* b) {
        float sum = 0;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    void load_data(const std::string& base_p, const std::string& query_p, const std::string& gt_p) {
        std::cout << "Loading Base...\n";
        std::ifstream in(base_p, std::ios::binary);
        if (!in.is_open()) { 
            std::cerr << "FATAL ERROR: Failed to open Base\n";
            exit(1); 
        }
        in.read((char*)&dim, 4);
        in.seekg(0, std::ios::end);
        num_points = in.tellg() / (4 + dim * 4);
        data = new float[num_points * dim];
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_points; ++i) {
            in.seekg(4, std::ios::cur);
            in.read((char*)(data + i * dim), dim * 4);
        }
        in.close();
        std::cout << "Loaded base: " << num_points << " points.\n";

        std::ifstream q_in(query_p, std::ios::binary);
        if (!q_in.is_open()) { exit(1); }
        int q_dim; q_in.read((char*)&q_dim, 4);
        q_in.seekg(0, std::ios::end);
        num_queries = q_in.tellg() / (4 + dim * 4);
        queries = new float[num_queries * dim];
        q_in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_queries; ++i) {
            q_in.seekg(4, std::ios::cur);
            q_in.read((char*)(queries + i * dim), dim * 4);
        }
        q_in.close();
        std::cout << "Loaded queries: " << num_queries << "\n";

        std::ifstream g_in(gt_p, std::ios::binary);
        if (!g_in.is_open()) { exit(1); }
        g_in.read((char*)&k_gt, 4);
        groundtruth = new uint32_t[num_queries * k_gt];
        g_in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num_queries; ++i) {
            g_in.seekg(4, std::ios::cur);
            g_in.read((char*)(groundtruth + i * k_gt), k_gt * 4);
        }
        g_in.close();
        
        graph.resize(num_points);
    }

    void crossing_prune_safe(uint32_t u, const std::vector<uint32_t>& candidates, uint32_t R, float alpha, std::vector<uint32_t>& out_edges) {
        std::vector<Neighbor> sorted;
        sorted.reserve(candidates.size());
        for(auto id : candidates) {
            if (id != u) sorted.push_back({id, compute_l2(data + u*dim, data + id*dim)});
        }
        std::sort(sorted.begin(), sorted.end());

        out_edges.clear();
        out_edges.reserve(R);
        for (auto& v : sorted) {
            bool occluded = false;
            for (uint32_t n : out_edges) {
                if (alpha * compute_l2(data + n*dim, data + v.id*dim) <= v.dist) {
                    occluded = true; break;
                }
            }
            if (!occluded) {
                out_edges.push_back(v.id);
                if (out_edges.size() >= R) break;
            }
        }
    }

    void search_impl(const float* q, uint32_t L, std::vector<uint32_t>& res, uint32_t seed, std::vector<uint8_t>& visited, std::vector<uint32_t>& visit_log) {
        std::vector<RetMeta> candidates;
        candidates.reserve(L + 100);

        uint32_t start = seed % num_points; 
        float d = compute_l2(q, data + start*dim);
        candidates.push_back({start, d, false});
        visited[start] = 1;
        visit_log.push_back(start);

        while (true) {
            int best_idx = -1;
            float min_d = 1e30;
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (!candidates[i].expanded && candidates[i].dist < min_d) {
                    min_d = candidates[i].dist;
                    best_idx = i;
                }
            }
            if (best_idx == -1) break;

            candidates[best_idx].expanded = true;
            uint32_t curr = candidates[best_idx].id;

            for (uint32_t nb : graph[curr]) {
                if (visited[nb]) continue;
                visited[nb] = 1;
                visit_log.push_back(nb);
                
                float nd = compute_l2(q, data + nb*dim);
                candidates.push_back({nb, nd, false});
            }

            std::sort(candidates.begin(), candidates.end());
            if (candidates.size() > L) candidates.resize(L);
        }

        for (uint32_t id : visit_log) visited[id] = 0;
        visit_log.clear();

        for (auto& c : candidates) res.push_back(c.id);
    }

    void build_index(uint32_t R, float alpha, uint32_t L_build) {
        std::cout << "\nStarting index build...\n";
        
        std::srand(42);
        for (size_t i = 0; i < num_points; ++i) {
            graph[i].reserve(R);
            for(uint32_t j=0; j<R; ++j) {
                uint32_t rn = std::rand() % num_points;
                if (rn != i) graph[i].push_back(rn);
            }
            std::sort(graph[i].begin(), graph[i].end());
            graph[i].erase(std::unique(graph[i].begin(), graph[i].end()), graph[i].end());
        }

        for (int pass = 0; pass < 2; ++pass) {
            std::cout << "  Pass " << pass + 1 << " / 2 ...\n";
            std::vector<std::vector<uint32_t>> next_graph(num_points);

            #pragma omp parallel
            {
                std::vector<uint8_t> visited(num_points, 0);
                std::vector<uint32_t> visit_log;
                visit_log.reserve(L_build * 2);

                #pragma omp for schedule(dynamic, 64)
                for (uint32_t i = 0; i < num_points; ++i) {
                    std::vector<uint32_t> pool;
                    search_impl(data + i*dim, L_build, pool, i, visited, visit_log);
                    for (auto nb : graph[i]) pool.push_back(nb);
                    std::sort(pool.begin(), pool.end());
                    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
                    
                    crossing_prune_safe(i, pool, R, alpha, next_graph[i]);
                }
            }
            
            for (uint32_t i = 0; i < num_points; ++i) {
                graph[i] = std::move(next_graph[i]);
            }
            
            std::vector<std::vector<uint32_t>> reverse_edges(num_points);
            for (uint32_t i = 0; i < num_points; ++i) {
                for (uint32_t nb : graph[i]) {
                    reverse_edges[nb].push_back(i);
                }
            }
            
            #pragma omp parallel for schedule(dynamic, 64)
            for (uint32_t i = 0; i < num_points; ++i) {
                std::vector<uint32_t> pool = graph[i];
                for (uint32_t rev : reverse_edges[i]) pool.push_back(rev);
                std::sort(pool.begin(), pool.end());
                pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
                
                if (pool.size() > R) {
                    std::vector<uint32_t> new_edges;
                    crossing_prune_safe(i, pool, R, alpha, new_edges);
                    graph[i] = std::move(new_edges);
                } else {
                    graph[i] = std::move(pool);
                }
            }
        }
        std::cout << "Index built.\n\n";
    }

    void test(uint32_t L) {
        double start = omp_get_wtime();
        size_t hit = 0;
        
        #pragma omp parallel
        {
            std::vector<uint8_t> visited(num_points, 0);
            std::vector<uint32_t> visit_log;
            visit_log.reserve(L * 2);

            #pragma omp for reduction(+:hit) schedule(dynamic, 16)
            for (uint32_t i = 0; i < num_queries; ++i) {
                std::vector<uint32_t> res;
                search_impl(queries + i*dim, L, res, 0, visited, visit_log);
                std::set<uint32_t> gt;
                for(int j=0; j<10; ++j) gt.insert(groundtruth[i*k_gt + j]);
                for(auto id : res) if(gt.count(id)) hit++;
            }
        }
        
        double time = omp_get_wtime() - start;
        std::cout << "L=" << L << " | Recall: " << (double)hit/(num_queries*10) << " | QPS: " << num_queries/time << '\n';
    }
};

int main(int argc, char** argv) {
    // 从命令行读取 R，如果没有提供参数，则默认使用 32
    uint32_t R = 32;
    if (argc > 1) {
        R = std::atoi(argv[1]);
    }

    std::string base_path  = "/home/cc/hpdic/AdaDisk/experiments/data/gist/gist_base.fvecs";
    std::string query_path = "/home/cc/hpdic/AdaDisk/experiments/data/gist/gist_query.fvecs";
    std::string gt_path    = "/home/cc/hpdic/AdaDisk/experiments/data/gist/gist_groundtruth.ivecs";

    std::cout << "=== CSPG Baseline Testing ===\n";
    std::cout << "[CONFIG] Target R = " << R << "\n";

    CSPG idx(base_path, query_path, gt_path, 960);
    
    // 使用传入的 R 进行建图
    idx.build_index(R, 1.2, 150);
    
    std::cout << "=== Testing Search Performance ===\n";
    for(int l : {50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000}) {
        idx.test(l);
    }

    return 0;
}

/* Example output:
(fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/CSPG$ g++ -O3 -march=native -fopenmp cspg_clean.cpp -o cspg_clean.bin

(fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/CSPG$ ./cspg_clean.bin 
=== CSPG Baseline Testing ===
[CONFIG] Target R = 32
Loading Base...
Loaded base: 1000000 points.
Loaded queries: 1000

Starting index build...
  Pass 1 / 2 ...
  Pass 2 / 2 ...
Index built.

=== Testing Search Performance ===
L=50 | Recall: 0.6603 | QPS: 8699.02
L=100 | Recall: 0.7579 | QPS: 3509.74
L=150 | Recall: 0.8086 | QPS: 2551.23
L=200 | Recall: 0.8411 | QPS: 2088.68
L=300 | Recall: 0.8801 | QPS: 1831.63
L=400 | Recall: 0.9003 | QPS: 1394.35
L=500 | Recall: 0.9148 | QPS: 1192.18
L=600 | Recall: 0.9272 | QPS: 1015.85
L=700 | Recall: 0.9354 | QPS: 931.188
L=800 | Recall: 0.9413 | QPS: 912.583
L=900 | Recall: 0.9474 | QPS: 724.203
L=1000 | Recall: 0.9517 | QPS: 738.2

(fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/CSPG$ ./cspg_clean.bin 48
=== CSPG Baseline Testing ===
[CONFIG] Target R = 48
Loading Base...
Loaded base: 1000000 points.
Loaded queries: 1000

Starting index build...
  Pass 1 / 2 ...
  Pass 2 / 2 ...
Index built.

=== Testing Search Performance ===
L=50 | Recall: 0.7821 | QPS: 5603.49
L=100 | Recall: 0.8669 | QPS: 2402.59
L=150 | Recall: 0.9012 | QPS: 2284.44
L=200 | Recall: 0.9238 | QPS: 1757.6
L=300 | Recall: 0.9513 | QPS: 1029.92
L=400 | Recall: 0.9643 | QPS: 968.19
L=500 | Recall: 0.9717 | QPS: 800.033
L=600 | Recall: 0.9766 | QPS: 793.674
L=700 | Recall: 0.9805 | QPS: 801.321
L=800 | Recall: 0.9832 | QPS: 614.258
L=900 | Recall: 0.986 | QPS: 566.277
L=1000 | Recall: 0.9874 | QPS: 527.08
(fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/CSPG$ 

*/