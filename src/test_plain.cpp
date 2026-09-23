// test_plain.cpp - plaintext HNSW recall check on a cached graph (no finalize needed)
// Used to separate "graph or ground-truth is broken" from "the HGI routing is broken".
#include "common.h"
#include "hnsw.h"
#include <omp.h>

using namespace eh;
static std::string arg(int argc, char** argv, const std::string& k, const std::string& def) {
    for (int i = 1; i + 1 < argc; i++) if (k == argv[i]) return argv[i + 1];
    return def;
}

int main(int argc, char** argv) {
    std::string base_raw = arg(argc, argv, "--base-raw", "");
    size_t nbase = (size_t)atoll(arg(argc, argv, "--nbase", "1000000").c_str());
    std::string qpath = arg(argc, argv, "--query", "");
    std::string gpath = arg(argc, argv, "--gt", "");
    std::string cache = arg(argc, argv, "--cache", "");
    int nq_max = atoi(arg(argc, argv, "--nq", "300").c_str());
    int k = atoi(arg(argc, argv, "--k", "10").c_str());
    int ef = atoi(arg(argc, argv, "--ef", "100").c_str());

    size_t nb, nq; int d;
    std::vector<float> xb;
    if (!base_raw.empty()) { xb = read_raw_u8_as_float(base_raw, nbase, 128); nb = nbase; d = 128; }
    else { fprintf(stderr, "need --base-raw\n"); return 1; }

    std::vector<float> xq;
    if (qpath.size() > 6 && qpath.compare(qpath.size() - 6, 6, ".bvecs") == 0)
        xq = read_bvecs_as_float(qpath, &nq, &d);
    else
        xq = read_fvecs(qpath, &nq, &d);

    size_t ng; int dgt;
    auto gt = read_ivecs(gpath, &ng, &dgt);

    HNSW g;
    if (!g.load(cache, xb.data())) { fprintf(stderr, "cache load failed\n"); return 1; }
    printf("[plain] graph n=%zu max_level=%d upper=%zu\n", g.n, g.max_level, g.count_upper(0));
    printf("[plain] links_.size()=%zu levels.size()=%zu blk_off.size()=%zu\n",
           g.links_.size(), g.levels.size(), g.blk_off.size());
    printf("[plain] blk_off[0..4]=%u %u %u %u %u   blk_off[mid]=%u  blk_off[n-1]=%u\n",
           g.blk_off[0], g.blk_off[1], g.blk_off[2], g.blk_off[3], g.blk_off[4],
           g.blk_off[g.n/2], g.blk_off[g.n-1]);
    printf("[plain] deg(0)=%u deg(1)=%u deg(mid)=%u deg(entry)=%u\n",
           g.deg(0,0), g.deg(1,0), g.deg((u32)(g.n/2),0), g.deg(g.entry_point,0));
    printf("[plain] prefix[0..2]=%u %u %u  M0=%d M=%d\n",
           g.prefix[0], g.prefix[1], g.prefix[2], g.M0, g.M);

    // ---- sanity: is the ground truth actually correct for this base?
    printf("[plain] GT id range: min=%d max=%d (must be < %zu)\n",
           (int)*std::min_element(gt.begin(), gt.begin() + 100 * dgt),
           (int)*std::max_element(gt.begin(), gt.begin() + 100 * dgt), g.n);

    int nq_use = std::min<int>(nq_max, (int)nq);
    long hits = 0, total = 0;
    double mq = 0;
    #pragma omp parallel for schedule(dynamic,8) reduction(+:hits,total,mq)
    for (int i = 0; i < nq_use; i++) {
        std::vector<u32> res;
        Timer t;
        g.search(xq.data() + (size_t)i * d, k, ef, res);
        mq += t.ms();
        int h = 0;
        for (u32 r : res) for (int j = 0; j < k; j++) if ((i32)r == gt[(size_t)i*dgt + j]) { h++; break; }
        hits += h; total += k;
    }
    printf("[plain] ef=%d  Recall@%d = %.4f  (%.1f us/query, %d threads)\n",
           ef, k, (double)hits / total, mq * 1000 / nq_use, omp_get_max_threads());

    // ---- structural sanity of the graph: layer-0 BFS coverage + degree stats
    {
        std::vector<u8> seen(g.n, 0);
        std::vector<u32> q; q.reserve(1u << 20);
        q.push_back(g.entry_point); seen[g.entry_point] = 1;
        size_t head = 0;
        while (head < q.size()) {
            u32 v = q[head++];
            u32 c = g.deg(v, 0);
            const u32* nb = g.nbrs(v, 0);
            for (u32 j = 0; j < c; j++) {
                u32 w = nb[j];
                if (w < g.n && !seen[w]) { seen[w] = 1; q.push_back(w); }
            }
        }
        size_t tot = 0, mx = 0, mn = (size_t)-1, zero = 0, oob = 0;
        for (size_t i = 0; i < g.n; i++) {
            u32 c = g.deg(i, 0);
            tot += c; mx = std::max<size_t>(mx, c); mn = std::min<size_t>(mn, c);
            if (!c) zero++;
            if (c > g.M0) oob++;
        }
        printf("[plain] L0 degree avg=%.1f min=%zu max=%zu zero=%zu  (M0=%d, over-cap=%zu)\n",
               (double)tot / g.n, mn, mx, zero, g.M0, oob);
        printf("[plain] L0 BFS from entry covers %zu / %zu = %.4f\n", q.size(), g.n,
               (double)q.size() / g.n);
        // does the entry point's neighbourhood actually look local?
        const float* pe = g.vec(g.entry_point);
        double d_nb = 0, d_rnd = 0; int cn = 0, cr = 0;
        for (u32 j = 0; j < g.deg(g.entry_point, 0); j++) {
            d_nb += l2_sqr(pe, g.vec(g.nbrs(g.entry_point,0)[j]), g.d); cn++;
        }
        for (size_t i = 0; i < 1000 && i < g.n; i++) { d_rnd += l2_sqr(pe, g.vec((u32)(i*7919u % g.n)), g.d); cr++; }
        printf("[plain] entry: avg dist to its neighbours = %.1f, to 1000 random nodes = %.1f\n",
               cn ? d_nb/cn : 0.0, cr ? d_rnd/cr : 0.0);
    }

    return 0;
}
