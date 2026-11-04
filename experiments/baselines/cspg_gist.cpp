// CSPG (official ANNS library, NeurIPS'24) on GIST1M: build once, search each efq once.
// Same parameters as the authors' overall/gist1m_vamana.cpp (Vamana subgraphs, 2 partitions, efc 128,
// alpha 1.2, dedup 0.5), with max degree R and thread count from the command line.
// Usage: cspg_gist <base.fvecs> <query.fvecs> <gt.ivecs> <R> <build_threads> <search_threads> <out.csv>
#include <fstream>
#include <iostream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/get_recall.hpp>
#include <utils/resize.hpp>
#include <utils/stimer.hpp>

using vdim_t = float;
using subgraph_t = anns::graph::DiskANN<vdim_t>;

int main(int argc, char **argv) {
  if (argc != 8) { std::cerr << "usage: base query gt R build_threads search_threads out.csv\n"; return 1; }
  const size_t R = std::stoul(argv[4]), TB = std::stoul(argv[5]), T = std::stoul(argv[6]), k = 10, part = 2;
  std::vector<vdim_t> base, queries;
  std::vector<id_t> gt;
  auto [nb, d0] = utils::LoadFromFile(base, argv[1]);
  auto [nq, d1] = utils::LoadFromFile(queries, argv[2]);
  auto [ng, d2] = utils::LoadFromFile(gt, argv[3]);
  auto nested = utils::Nest(std::move(queries), nq, d1);
  utils::STimer bt, qt;
  auto cspg = std::make_unique<anns::graph::RandomPartitionGraph<vdim_t, subgraph_t>>(d0, part);
  bt.Start();
  cspg->BuildIndex(base, TB, {(float)R, 1.2f, 128.0f, 0.5f});
  bt.Stop();
  std::ofstream csv(argv[7]);
  csv << "R,efq,build_s,qps,recall\n";
  for (size_t efq : {10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800}) {
    std::vector<std::vector<id_t>> knn;
    qt.Reset();
    qt.Start();
    cspg->GetTopkNNParallel(nested, k, T, 1, efq, knn);
    qt.Stop();
    double recall = utils::GetRecall(k, d2, gt, knn);
    csv << R << "," << efq << "," << bt.GetTime() << "," << nq / qt.GetTime() << "," << recall << std::endl;
    std::cout << "R" << R << " efq " << efq << " qps " << nq / qt.GetTime() << " recall " << recall << std::endl;
  }
  return 0;
}
