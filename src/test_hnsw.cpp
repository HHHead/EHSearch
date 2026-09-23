// test_hnsw.cpp - validate the self-written HNSW against SIFT-1M ground truth
#include "common.h"
#include "hnsw.h"
#include <omp.h>

using namespace eh;

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "./data/sift";
    int M = argc > 2 ? atoi(argv[2]) : 32;
    int efC = argc > 3 ? atoi(argv[3]) : 64;
    int efS = argc > 4 ? atoi(argv[4]) : 100;
    int k = 20;
    print_cpu_caps();
    printf("threads=%d\n", omp_get_max_threads());

    size_t nb, nq; int d, dq;
    auto xb = read_fvecs(root + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(root + "/sift_query.fvecs", &nq, &dq);
    size_t ngt; int dgt;
    auto gt = read_ivecs(root + "/sift_groundtruth.ivecs", &ngt, &dgt);
    printf("base %zu x %d, query %zu x %d, gt %zu x %d, rss=%.0fMB\n",
           nb, d, nq, dq, ngt, dgt, rss_mb());

    HNSW g;
    Timer t;
    g.build(xb.data(), nb, d, M, efC, 42, true);
    printf("build %.1fs\n", t.sec());
    g.finalize();

    // plaintext search quality
    for (int efS : {32, 50, 100, 200, 400}) {
        double hits = 0; long total = 0;
        Timer tq;
        #pragma omp parallel for schedule(dynamic, 32) reduction(+:hits,total)
        for (long i = 0; i < (long)nq; i++) {
            std::vector<u32> res;
            g.search(xq.data() + i * d, k, efS, res);
            int h = 0;
            for (u32 r : res) for (int j = 0; j < k; j++) if ((i32)r == gt[i * dgt + j]) { h++; break; }
            hits += h; total += k;
        }
        printf("plaintext HNSW  M=%d efC=%d efSearch=%3d : Recall@%d = %.4f  (%.1f us/query, %d threads)\n",
               M, efC, efS, k, hits / total, tq.ms() * 1000 / nq, omp_get_max_threads());
        fflush(stdout);
    }
    return 0;
}
