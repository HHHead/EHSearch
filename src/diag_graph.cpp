// diag_graph.cpp - diagnose HNSW layer-0 connectivity / reachability
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
    int M = atoi(arg(argc, argv, "--M", "32").c_str());
    int efC = atoi(arg(argc, argv, "--efC", "64").c_str());

    size_t nb, nq; int d, dgt;
    auto xb = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &d);
    size_t ng; auto gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &ng, &dgt);

    HNSW g;
    Timer t;
    if (!g.load(cache, xb.data())) {
        g.build(xb.data(), nb, d, M, efC, 42, true);
        g.save(cache);
    }
    printf("graph ready %.1fs\n", t.sec());
    g.finalize();

    // ---- degree stats at layer 0
    {
        size_t tot = 0, mx = 0, mn = (size_t)-1, zero = 0;
        std::vector<size_t> hist(80, 0);
        for (u32 i = 0; i < g.n; i++) {
            u32 c = g.deg(i, 0); tot += c; mx = std::max<size_t>(mx, c); mn = std::min<size_t>(mn, c);
            if (c == 0) zero++;
            hist[std::min<size_t>(c, 79)]++;
        }
        printf("[L0 degree] avg=%.1f min=%zu max=%zu zero_deg=%zu\n", (double)tot / g.n, mn, mx, zero);
    }
    // ---- reachability BFS from entry point
    {
        std::vector<u8> seen(g.n, 0);
        std::vector<u32> q; q.reserve(g.n);
        q.push_back(g.entry_point); seen[g.entry_point] = 1;
        size_t head = 0;
        while (head < q.size()) {
            u32 v = q[head++];
            for (u64 k = g.l0_off[v]; k < g.l0_off[v + 1]; k++) {
                u32 w = g.l0_nbr[k];
                if (!seen[w]) { seen[w] = 1; q.push_back(w); }
            }
        }
        printf("[L0 reachability] reachable from entry = %zu / %zu = %.4f  (UNREACHABLE %zu)\n",
               q.size(), g.n, (double)q.size() / g.n, g.n - q.size());
    }
    // ---- how many queries' true top-20 are all reachable
    long tot20 = 0, reach20 = 0, tot1 = 0, reach1 = 0;
    std::vector<u8> seen(g.n, 0);
    for (int i = 0; i < 200 && i < (int)nq; i++) {
        for (int j = 0; j < 20; j++) {
            u32 id = (u32)gt[i * dgt + j];
            tot20++; if (seen[id]) reach20++;
            if (j == 0) { tot1++; if (seen[id]) reach1++; }
        }
    }
    printf("[Reach of true NN] top1 %.4f  top20-any %.4f  (all-reachable check below)\n",
           (double)reach1 / tot1, (double)reach20 / tot20);

    // ---- grep by plaintext search at several ef
    for (int ef : {1, 10, 50, 100, 400, 1000}) {
        long hits = 0, total = 0;
        int nqi = std::min<int>(2000, nq);
        #pragma omp parallel for schedule(dynamic,16) reduction(+:hits,total)
        for (int i = 0; i < nqi; i++) {
            std::vector<u32> res; g.search(xq.data() + (size_t)i * d, 20, ef, res);
            int h = 0;
            for (u32 r : res) for (int j = 0; j < 20; j++) if ((i32)r == gt[i * dgt + j]) { h++; break; }
            hits += h; total += 20;
        }
        printf("[search] ef=%4d Recall@20=%.4f\n", ef, (double)hits / total);
    }
    return 0;
}
