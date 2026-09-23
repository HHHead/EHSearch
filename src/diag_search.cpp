// diag_search.cpp - isolate the recall plateau on the cached 1M graph
#include "common.h"
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
    std::string cache = arg(argc, argv, "--cache",
        "./build/hnsw_M32_efC64.bin");
    size_t nb, nq; int d, dgt;
    auto xb = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &d);
    size_t ng; auto gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &ng, &dgt);
    HNSW g;
    if (!g.load(cache, xb.data())) {
        printf("[HNSW] cache miss, building...\n"); fflush(stdout);
        g.build(xb.data(), nb, d, 32, 64, 42, true);
        g.save(cache);
    }
    g.finalize();
    const int NQ = 500;

    // ---- (A) sanity: does brute force reproduce the official ground truth?
    {
        long hits = 0, total = 0;
        Timer t;
        #pragma omp parallel num_threads(12) reduction(+:hits,total)
        {
            std::vector<std::pair<float,int>> v(nb);
            #pragma omp for schedule(dynamic,4)
            for (int i = 0; i < NQ; i++) {
                const float* q = xq.data() + (size_t)i*d;
                for (size_t j = 0; j < nb; j++) v[j] = {l2_sqr(q, xb.data()+j*(size_t)d, d), (int)j};
                std::partial_sort(v.begin(), v.begin()+20, v.end());
                int h = 0;
                for (int a = 0; a < 20; a++)
                    for (int b = 0; b < 20; b++) if (v[a].second == gt[(size_t)i*dgt+b]) { h++; break; }
                hits += h; total += 20;
            }
        }
        printf("[sanity] brute-force vs official GT: Recall@20 = %.4f  (%.1fs)\n",
               (double)hits/total, t.sec());
    }

    // ---- (B) standard search, varying ef
    for (int ef : {50, 100, 400}) {
        long hits = 0, total = 0;
        #pragma omp parallel for schedule(dynamic,16) reduction(+:hits,total)
        for (int i = 0; i < NQ; i++) {
            std::vector<u32> res; g.search(xq.data()+(size_t)i*d, 20, ef, res);
            int h = 0; for (u32 r : res) for (int j = 0; j < 20; j++)
                if ((i32)r == gt[(size_t)i*dgt+j]) { h++; break; }
            hits += h; total += 20;
        }
        printf("[search] ef=%4d Recall@20 = %.4f\n", ef, (double)hits/total);
    }
    return 0;
}
