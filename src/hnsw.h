// hnsw.h - self-contained HNSW (build + plaintext search + layer-0 CSR export)
#pragma once
#include "common.h"
#include "simd.h"
#include <random>
#include <queue>
#include <mutex>

namespace eh {

// Chunked file I/O.  A single ifstream read/write of more than 2 GB is not
// reliable on Windows/MinGW: the call can return short and silently zero-fill
// the tail of the destination buffer.  This was the root cause of a 10M-node
// graph loading with an almost entirely empty layer-0 (avg degree 0.0,
// Recall@10 = 0.0000) while the on-disk file was perfectly intact.
static constexpr size_t IO_CHUNK = 1u << 28;   // 256 MB

inline bool read_chunked(std::ifstream& f, void* dst, size_t bytes) {
    u8* p = (u8*)dst;
    size_t left = bytes;
    while (left) {
        const size_t k = std::min(left, IO_CHUNK);
        f.read((char*)p, (std::streamsize)k);
        if ((size_t)f.gcount() != k) return false;
        p += k; left -= k;
    }
    return true;
}

inline void write_chunked(std::ofstream& f, const void* src, size_t bytes) {
    const u8* p = (const u8*)src;
    size_t left = bytes;
    while (left) {
        const size_t k = std::min(left, IO_CHUNK);
        f.write((const char*)p, (std::streamsize)k);
        p += k; left -= k;
    }
}

class HNSW {
public:
    int d = 0, M = 32, M0 = 64, efC = 64;
    double mL = 1.0 / std::log(32.0);
    size_t n = 0;
    const float* data = nullptr;

    int max_level = -1;
    u32 entry_point = 0;

    std::vector<i32> levels;          // 0-based top layer index per node
    std::vector<u32> blk_off;         // n+1 offsets into links_
    std::vector<u32> links_;          // [count][nbr0..count-1] per (node,layer), cap = maxM[l]+1
    std::vector<u32> maxM;            // capacity per layer
    std::vector<u32> prefix;          // cumulative caps (in units of slots) -> prefix[l] = sum_{k<l}(maxM[k]+1)

    // layer-0 CSR (built by finalize())
    std::vector<u64> l0_off;
    std::vector<u32> l0_nbr;
    bool finalized = false;

    // construction knobs.
    // IMPORTANT: leaving keep_pruned enabled fills every adjacency list to the
    // capacity with the closest discarded candidates, which degenerates the
    // graph into a k-NN graph: measured layer-0 BFS coverage from the entry
    // point dropped to 0.47 and Recall@20 (SIFT-1M, ef=100) to 0.88.  Keeping
    // only the RNG-diverse neighbours (keep_pruned = false) yields avg degree
    // ~20, 100% coverage and Recall@20 = 0.972, matching faiss HNSW(M=32,
    // efConstruction=64) = 0.9717.
    bool keep_pruned_sel = false;
    bool keep_pruned_shrink = false;
    int m0_factor = 2;
    void set_keep_pruned(bool v) { keep_pruned_sel = v; keep_pruned_shrink = v; }
    void set_keep_pruned2(bool sel, bool shrink) { keep_pruned_sel = sel; keep_pruned_shrink = shrink; }
    void set_m0_factor(int f) { m0_factor = f; }

    inline u32 slot_base(u32 node, int l) const { return blk_off[node] + prefix[l]; }
    inline u32 deg(u32 node, int l) const { return links_[slot_base(node, l)]; }
    inline const u32* nbrs(u32 node, int l) const { return links_.data() + slot_base(node, l) + 1; }
    inline u32* nbrs_mut(u32 node, int l) { return links_.data() + slot_base(node, l) + 1; }
    inline const float* vec(u32 i) const { return data + (size_t)i * d; }

    void init(int d_, int M_, int efC_, int seed = 42) {
        d = d_; M = M_; M0 = m0_factor * M_; efC = efC_; mL = 1.0 / std::log((double)M_);
        rng.seed(seed);
        levels.clear(); blk_off.clear(); links_.clear();
        max_level = -1; n = 0;
        maxM.clear(); prefix.clear();
        // layers 0..31 supported
        maxM.resize(32); prefix.resize(33);
        maxM[0] = M0; for (int l = 1; l < 32; l++) maxM[l] = M;
        prefix[0] = 0; for (int l = 0; l < 32; l++) prefix[l + 1] = prefix[l] + maxM[l] + 1;
        finalized = false;
    }

    int random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        if (r <= 0) r = 1e-12;
        int lv = (int)(-std::log(r) * mL);
        if (lv > 31) lv = 31;
        return lv;
    }

    // ---------------------------------------------------------------- build
    void add_point(u32 id) {
        int lv = random_level();
        if (n == 0) {
            levels.push_back(lv);
            blk_off.push_back((u32)links_.size());
            links_.resize(links_.size() + prefix[lv + 1], 0);
            n = 1; max_level = lv; entry_point = 0;
            return;
        }
        // allocate block for the new node
        u32 myblk = (u32)links_.size();
        links_.resize(links_.size() + prefix[lv + 1], 0);
        if ((u32)levels.size() < id) levels.resize(id, 0);
        levels.push_back(lv);
        blk_off.push_back(myblk);
        n = id + 1;

        const float* q = vec(id);

        // phase 1: greedy descent above the node's level
        u32 cur = entry_point;
        float curd = l2_sqr(q, vec(cur), d);
        for (int l = max_level; l > lv; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                u32 c = deg(cur, l);
                const u32* nb = nbrs(cur, l);
                for (u32 j = 0; j < c; j++) {
                    float dd = l2_sqr(q, vec(nb[j]), d);
                    if (dd < curd) { curd = dd; cur = nb[j]; changed = true; }
                }
            }
        }

        // phase 2: connect layer by layer
        std::vector<std::pair<float, u32>> W;
        int start = std::min(lv, max_level);
        for (int l = start; l >= 0; l--) {
            search_layer(q, cur, curd, l, efC, W);
            u32 cap = (l == 0) ? (u32)M0 : (u32)M;
            std::vector<u32> sel;
            select_heuristic<true>(q, W, cap, sel, keep_pruned_sel);

            // out-edges: new node -> selected
            set_list(id, l, sel);
            // in-edges: selected -> new node
            for (u32 nb : sel) {
                add_link(nb, l, id, l == 0);
            }
            if (!W.empty()) { cur = W[0].second; curd = W[0].first; }
        }

        if (lv > max_level) { max_level = lv; entry_point = id; }
    }

    void build(const float* data_, size_t n_, int d_, int M_, int efC_, int seed = 42,
               bool verbose = true) {
        init(d_, M_, efC_, seed);
        data = data_;
        levels.reserve(n_); blk_off.reserve(n_ + 1);
        links_.reserve(n_ * (size_t)(prefix[1]));
        Timer t;
        for (u32 i = 0; i < n_; i++) {
            add_point(i);
            if (verbose && n_ >= 100000 && (i + 1) % (n_ / 10) == 0) {
                printf("  [HNSW] %5.1f%%  (%.1fs, rss %.0f MB)\n",
                       100.0 * (i + 1) / n_, t.sec(), rss_mb());
                fflush(stdout);
            }
        }
        if (verbose) printf("  [HNSW] build done: %.1fs, levels>0 = %zu (%.3f%%), max_level=%d\n",
                            t.sec(), count_upper(0), 100.0 * count_upper(0) / n_, max_level);
    }

    size_t count_upper(int thr) const {  // count nodes with levels[i] > thr
        size_t c = 0; for (i32 lv : levels) if (lv > thr) c++; return c;
    }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "[HNSW] cannot write %s\n", path.c_str()); return; }
        u64 magic = 0x484E535730303031ull;
        f.write((const char*)&magic, 8);
        i64 n64 = (i64)n; int ml = max_level; u32 ep = entry_point;
        f.write((const char*)&d, 4); f.write((const char*)&M, 4);
        f.write((const char*)&M0, 4); f.write((const char*)&efC, 4);
        f.write((const char*)&n64, 8); f.write((const char*)&ml, 4);
        f.write((const char*)&ep, 4);
        f.write((const char*)levels.data(), (size_t)n * 4);
        f.write((const char*)blk_off.data(), (size_t)n * 4);   // must match load(): exactly n entries
        u64 ls = links_.size();
        f.write((const char*)&ls, 8);
        write_chunked(f, links_.data(), links_.size() * 4);
        f.flush();
        f.close();
        printf("[HNSW] cached to %s (%.0f MB)\n", path.c_str(), file_size(path) / 1048576.0);
    }
    bool load(const std::string& path, const float* data_) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        u64 magic = 0; f.read((char*)&magic, 8);
        if (magic != 0x484E535730303031ull) { fprintf(stderr, "[HNSW] bad cache magic\n"); return false; }
        i64 n64 = 0; int ml = 0; u32 ep = 0;
        f.read((char*)&d, 4); f.read((char*)&M, 4); f.read((char*)&M0, 4); f.read((char*)&efC, 4);
        f.read((char*)&n64, 8); f.read((char*)&ml, 4); f.read((char*)&ep, 4);
        n = (size_t)n64; max_level = ml; entry_point = ep;
        mL = 1.0 / std::log((double)M);
        maxM.resize(32); prefix.resize(33);
        maxM[0] = M0; for (int l = 1; l < 32; l++) maxM[l] = M;
        prefix[0] = 0; for (int l = 0; l < 32; l++) prefix[l + 1] = prefix[l] + maxM[l] + 1;
        levels.resize(n); blk_off.resize(n);
        f.read((char*)levels.data(), n * 4);
        f.read((char*)blk_off.data(), n * 4);
        u64 ls = 0; f.read((char*)&ls, 8);
        links_.resize(ls);
        if (!read_chunked(f, links_.data(), ls * 4)) {
            fprintf(stderr, "[HNSW] failed to read adjacency array (%llu B)\n",
                    (unsigned long long)(ls * 4));
            return false;
        }
        data = data_;
        printf("[HNSW] loaded cache %s: n=%zu M=%d efC=%d max_level=%d upper=%zu (%.3f%%)\n",
               path.c_str(), n, M, efC, max_level, count_upper(0), 100.0 * count_upper(0) / n);
        return true;
    }

    void finalize() {
        l0_off.assign(n + 1, 0);
        size_t tot = 0;
        for (u32 i = 0; i < n; i++) { l0_off[i] = tot; tot += deg(i, 0); }
        l0_off[n] = tot;
        l0_nbr.resize(tot);
        for (u32 i = 0; i < n; i++) {
            u32 c = deg(i, 0); const u32* nb = nbrs(i, 0);
            memcpy(l0_nbr.data() + l0_off[i], nb, (size_t)c * 4);
        }
        finalized = true;
    }

    // ---------------------------------------------------------------- search
    struct Cand { float d; u32 id; };
    struct CmpMax { bool operator()(const Cand& a, const Cand& b) const { return a.d < b.d; } };  // max-heap
    struct CmpMin { bool operator()(const Cand& a, const Cand& b) const { return a.d > b.d; } };  // min-heap

    // plaintext search over the whole hierarchy (validation / baseline)
    void search(const float* q, int k, int ef, std::vector<u32>& out) const {
        u32 cur = entry_point;
        float curd = l2_sqr(q, vec(cur), d);
        for (int l = max_level; l >= 1; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                u32 c = deg(cur, l); const u32* nb = nbrs(cur, l);
                for (u32 j = 0; j < c; j++) {
                    float dd = l2_sqr(q, vec(nb[j]), d);
                    if (dd < curd) { curd = dd; cur = nb[j]; changed = true; }
                }
            }
        }
        std::vector<std::pair<float, u32>> W;
        search_layer(q, cur, curd, 0, std::max(ef, k), W);
        out.clear();
        for (size_t i = 0; i < W.size() && (int)out.size() < k; i++) out.push_back(W[i].second);
    }

    // greedy search restricted to layer l (used by client-side local routing on upper layers)
    void greedy_layer(const float* q, u32& cur, float& curd, int l) const {
        bool changed = true;
        while (changed) {
            changed = false;
            u32 c = deg(cur, l); const u32* nb = nbrs(cur, l);
            for (u32 j = 0; j < c; j++) {
                float dd = l2_sqr(q, vec(nb[j]), d);
                if (dd < curd) { curd = dd; cur = nb[j]; changed = true; }
            }
        }
    }

    // best-first exploration of a single layer, returns up to ef sorted ascending
    void search_layer(const float* q, u32 ep, float epd, int l, int ef,
                      std::vector<std::pair<float, u32>>& out) const {
        static thread_local std::vector<u32> vis_stamp;
        static thread_local u32 epoch = 0;
        if (vis_stamp.size() < n) vis_stamp.resize(n, 0);
        epoch++; if (epoch == 0) { std::fill(vis_stamp.begin(), vis_stamp.end(), 0); epoch = 1; }

        std::priority_queue<Cand, std::vector<Cand>, CmpMax> top;   // candidates (min by d on top? CmpMax => max on top) -> we want min, so use CmpMin
        std::priority_queue<Cand, std::vector<Cand>, CmpMin> cand;
        std::priority_queue<Cand, std::vector<Cand>, CmpMax> res;
        cand.push({epd, ep}); res.push({epd, ep}); vis_stamp[ep] = epoch;

        while (!cand.empty()) {
            Cand c = cand.top(); cand.pop();
            if (res.size() >= (size_t)ef && c.d > res.top().d) break;
            u32 degc = deg(c.id, l); const u32* nb = nbrs(c.id, l);
            for (u32 j = 0; j < degc; j++) {
                u32 v = nb[j];
                if (vis_stamp[v] == epoch) continue;
                vis_stamp[v] = epoch;
                float dd = l2_sqr(q, vec(v), d);
                if (res.size() < (size_t)ef || dd < res.top().d) {
                    cand.push({dd, v}); res.push({dd, v});
                    if (res.size() > (size_t)ef) res.pop();
                }
            }
        }
        out.clear(); out.reserve(res.size());
        while (!res.empty()) { out.push_back({res.top().d, res.top().id}); res.pop(); }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.first < b.first; });
    }

private:
    std::mt19937_64 rng;

    inline void set_list(u32 node, int l, const std::vector<u32>& lst) {
        u32 base = slot_base(node, l);
        u32 cap = (l == 0) ? (u32)M0 : (u32)M;
        u32 c = (u32)std::min<size_t>(lst.size(), cap);
        links_[base] = c;
        memcpy(links_.data() + base + 1, lst.data(), c * 4);
    }

    inline void add_link(u32 node, int l, u32 target, bool is_l0) {
        u32 base = slot_base(node, l);
        u32 cnt = links_[base];
        u32 cap = (l == 0) ? (u32)M0 : (u32)M;
        u32* nb = links_.data() + base + 1;
        for (u32 j = 0; j < cnt; j++) if (nb[j] == target) return;
        if (cnt < cap) { nb[cnt] = target; links_[base] = cnt + 1; return; }
        // over capacity -> shrink with heuristic
        std::vector<std::pair<float, u32>> cand;
        cand.reserve(cnt + 1);
        const float* pv = vec(node);
        for (u32 j = 0; j < cnt; j++) cand.push_back({l2_sqr(pv, vec(nb[j]), d), nb[j]});
        cand.push_back({l2_sqr(pv, vec(target), d), target});
        std::vector<u32> sel;
        // faiss keeps the pruned (closest discarded) connections so that the degree
        // stays at M/M0 -- using keep_pruned=false here measurably lowers recall.
        select_heuristic<true>(pv, cand, cap, sel, keep_pruned_shrink);
        set_list(node, l, sel);
    }

    // Algorithm 4 style heuristic neighbour selection.
    // keep_pruned=true fills up to cap with the closest discarded candidates.
    template <bool KP_UNUSED>
    void select_heuristic(const float* q, std::vector<std::pair<float, u32>>& cand,
                          u32 cap, std::vector<u32>& out, bool keep_pruned) const {
        out.clear();
        if (cand.empty()) return;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::pair<float, u32>> discarded;
        for (auto& c : cand) {
            if (out.size() >= cap) break;
            bool good = true;
            const float* cv = vec(c.second);
            for (u32 r : out) {
                if (l2_sqr(cv, vec(r), d) < c.first) { good = false; break; }
            }
            if (good) out.push_back(c.second);
            else discarded.push_back(c);
        }
        if (keep_pruned && out.size() < cap) {
            // fill from the closest discarded
            for (auto& c : discarded) {
                if (out.size() >= cap) break;
                out.push_back(c.second);
            }
        }
    }
};

} // namespace eh
