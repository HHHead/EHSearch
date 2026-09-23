// tune_hnsw.cpp - fast ablation of HNSW construction settings on a subset
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
    int N = atoi(arg(argc, argv, "--n", "100000").c_str());
    int NQ = atoi(arg(argc, argv, "--nq", "300").c_str());
    size_t nb, nq; int d, dgt;
    auto xball = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &d);
    auto gtfull = read_ivecs(dir + "/sift_groundtruth.ivecs", &nq, &dgt);
    std::vector<float> xb(xball.begin(), xball.begin() + (size_t)N * d);

    // brute-force ground truth on the subset
    std::vector<i32> gt((size_t)NQ * 20);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < NQ; i++) {
        std::vector<std::pair<float, int>> v(N);
        const float* q = xq.data() + (size_t)i * d;
        for (int j = 0; j < N; j++) v[j] = {l2_sqr(q, xb.data() + (size_t)j * d, d), j};
        std::partial_sort(v.begin(), v.begin() + 20, v.end());
        for (int j = 0; j < 20; j++) gt[(size_t)i * 20 + j] = v[j].second;
    }
    printf("[gt] brute force done (N=%d, NQ=%d)\n", N, NQ);
    fflush(stdout);

    struct Cfg { int M, efC, kpsel, kpshrink, m0f; };
    std::vector<Cfg> cfgs = {
        {32, 64, 0, 0, 2}, {32, 64, 1, 1, 2}, {32, 100, 0, 0, 2},
        {32, 100, 1, 1, 2}, {32, 150, 1, 1, 2},
    };
    for (auto c : cfgs) {
        HNSW g;
        g.set_keep_pruned2(c.kpsel != 0, c.kpshrink != 0);
        g.set_m0_factor(c.m0f);
        Timer t;
        g.build(xb.data(), N, d, c.M, c.efC, 42, false);
        double bt = t.sec();
        g.finalize();
        // degree + connectivity diagnostics
        size_t tot = 0, mx = 0, mn = (size_t)-1;
        for (u32 i = 0; i < g.n; i++) { u32 cc = g.deg(i,0); tot += cc;
            mx = std::max<size_t>(mx,cc); mn = std::min<size_t>(mn,cc); }
        std::vector<u8> seen(g.n, 0); std::vector<u32> q; q.reserve(g.n);
        q.push_back(g.entry_point); seen[g.entry_point] = 1; size_t head = 0;
        while (head < q.size()) { u32 v = q[head++];
            for (u64 k = g.l0_off[v]; k < g.l0_off[v+1]; k++) { u32 w = g.l0_nbr[k];
                if (!seen[w]) { seen[w] = 1; q.push_back(w); } } }
        double best = 0;
        std::string detail;
        for (int ef : {50, 100, 200}) {
            long hits = 0, total = 0;
            #pragma omp parallel for schedule(dynamic, 8) reduction(+:hits,total)
            for (int i = 0; i < NQ; i++) {
                std::vector<u32> res; g.search(xq.data() + (size_t)i * d, 20, ef, res);
                int h = 0;
                for (u32 r : res) for (int j = 0; j < 20; j++) if ((i32)r == gt[(size_t)i*20+j]) { h++; break; }
                hits += h; total += 20;
            }
            double rc = (double)hits / total;
            detail += " ef" + std::to_string(ef) + "=" + fmt(rc, 4);
            if (rc > best) best = rc;
        }
        printf("M=%2d efC=%3d kpSel=%d kpShrink=%d M0=%dM : %5.1fs | deg %.1f [%zu,%zu] | "
               "BFS %.4f |%s\n",
               c.M, c.efC, c.kpsel, c.kpshrink, c.m0f, bt, (double)tot/g.n, mn, mx,
               (double)q.size()/g.n, detail.c_str());
        fflush(stdout);
    }
    return 0;
}
