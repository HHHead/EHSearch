// tune_scale.cpp - scale sweep of graph connectivity for the self-written HNSW
#include "common.h"
#include "simd.h"
#include "hnsw.h"
#include <omp.h>

using namespace eh;
static std::string arg(int argc, char** argv, const std::string& k, const std::string& def) {
    for (int i = 1; i + 1 < argc; i++) if (k == argv[i]) return argv[i + 1];
    return def;
}

int main(int argc, char** argv) {
    std::string dir = arg(argc, argv, "--data",
        "./data/sift");
    int M = atoi(arg(argc, argv, "--M", "32").c_str());
    int efC = atoi(arg(argc, argv, "--efC", "64").c_str());
    int kp = atoi(arg(argc, argv, "--kp", "1").c_str());
    int NQ = atoi(arg(argc, argv, "--nq", "500").c_str());

    size_t nb, nq; int d, dgt;
    auto xb = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &d);
    auto gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &nq, &dgt);

    for (int N : {100000, 300000, 1000000}) {
        if (N > (int)nb) break;
        HNSW g; g.set_keep_pruned(kp != 0);
        Timer t;
        g.build(xb.data(), N, d, M, efC, 42, false);
        double bt = t.sec();
        g.finalize();
        // degree stats
        size_t tot = 0, mx = 0, mn = (size_t)-1, zero = 0;
        for (u32 i = 0; i < g.n; i++) { u32 c = g.deg(i, 0); tot += c; mx = std::max<size_t>(mx,c);
                                        mn = std::min<size_t>(mn,c); if (!c) zero++; }
        // BFS coverage from entry point
        std::vector<u8> seen(g.n, 0); std::vector<u32> q; q.reserve(g.n);
        q.push_back(g.entry_point); seen[g.entry_point] = 1; size_t head = 0;
        while (head < q.size()) { u32 v = q[head++];
            for (u64 k = g.l0_off[v]; k < g.l0_off[v+1]; k++) { u32 w = g.l0_nbr[k];
                if (!seen[w]) { seen[w] = 1; q.push_back(w); } } }
        double cov = (double)q.size() / g.n;
        // recall
        std::string det;
        for (int ef : {50, 100, 400}) {
            long hits = 0, total = 0;
            #pragma omp parallel for schedule(dynamic,16) reduction(+:hits,total)
            for (int i = 0; i < NQ; i++) {
                std::vector<u32> res; g.search(xq.data()+(size_t)i*d, 20, ef, res);
                int h = 0; for (u32 r : res) for (int j = 0; j < 20; j++)
                    if ((i32)r == gt[(size_t)i*dgt+j]) { h++; break; }
                hits += h; total += 20;
            }
            det += " ef" + std::to_string(ef) + "=" + fmt((double)hits/total, 4);
        }
        printf("N=%7d M=%d efC=%d kp=%d | %.0fs | deg avg=%.1f min=%zu max=%zu zero=%zu | "
               "BFS from entry covers %.4f | maxL=%d upper=%zu |%s\n",
               N, M, efC, kp, bt, (double)tot/g.n, mn, mx, zero, cov, g.max_level,
               g.count_upper(0), det.c_str());
        fflush(stdout);
    }
    return 0;
}
