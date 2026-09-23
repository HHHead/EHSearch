// hgi.h - Topology-Distilled Index (HGI) + Group-Scoped Multi-Index
#pragma once
#include "common.h"
#include "hnsw.h"
#include "crypto.h"
#include <omp.h>

namespace eh {

// ------------------------------------------------------------------ HGI groups
struct HGI {
    int B = 1500;
    int anchor_level = 1;                 // anchors = nodes with hnsw_level >= anchor_level
    std::vector<u32> anchor;              // group g -> anchor node id
    std::vector<u64> off;                 // G+1 offsets into members
    std::vector<u32> members;             // G * B member ids
    std::vector<i32> node2group;          // node id -> group index (-1 if not an anchor)
    double coverage = 0.0;                // distinct covered nodes / N
    double replication = 0.0;             // G*B / N
    size_t covered = 0;
    double build_sec = 0.0;

    size_t G() const { return anchor.size(); }

    // BFS along layer-0 edges from each anchor until the group holds B nodes.
    // Boundary nodes end up in several groups (overlapping padding); groups that
    // cannot be filled from a connected component are padded with the lowest
    // still-unused ids, as in the reference implementation.
    //
    // `complete_` enables the coverage-completion pass.  Increasing B alone does
    // NOT reach 100% coverage: once B exceeds the reach of the BFS the groups fill
    // straight from the giant component, padding stops firing, and every node
    // outside that component stays ungrouped.  Measured: coverage converges to the
    // giant-component fraction (GIST-1M 98.48%, SIFT-1M / arxiv768 / dbpedia1536
    // 100.00%).  The completion pass rewrites *duplicated* padding slots with the
    // still-uncovered ids, which guarantees coverage == 1.0 at zero extra storage.
    void build(const HNSW& g, int B_, int anchor_level_, bool verbose = true,
               bool complete_ = false, u32* set_aside = nullptr) {
        B = B_;
        anchor_level = anchor_level_;
        Timer t;
        const size_t N = g.n;
        anchor.clear();
        for (u32 i = 0; i < N; i++) if (g.levels[i] >= anchor_level) anchor.push_back(i);
        const size_t G = anchor.size();
        off.assign(G + 1, 0);
        members.assign(G * (size_t)B, 0);

        std::vector<u32> stamp(N, 0);
        size_t covered_cnt = 0;

        #pragma omp parallel
        {
            std::vector<u32> q;
            std::vector<u8> local_seen(N, 0);   // per-thread membership bitmap
            std::vector<u32> touched;
            // Rotating fill cursor, partitioned per thread.  Padding used to scan
            // from index 0 for every group, which is O(G*N) and pathological at
            // large N; it also read past the end of `local_seen` once every node
            // had already been covered (out-of-bounds -> crash at 10M).
            const int tid = omp_get_thread_num();
            const int ntd = omp_get_num_threads();
            u32 cursor = (u32)(((size_t)N / (size_t)ntd) * (size_t)tid);
            if (cursor >= N) cursor = 0;
            #pragma omp for schedule(dynamic, 8)
            for (long long gi = 0; gi < (long long)G; gi++) {
                u32 a = anchor[gi];
                u32* mem = members.data() + (size_t)gi * B;
                q.clear(); touched.clear();
                q.push_back(a);
                local_seen[a] = 1; touched.push_back(a);
                size_t cnt = 0;
                size_t head = 0;
                while (head < q.size() && cnt < (size_t)B) {
                    u32 cur = q[head++];
                    mem[cnt++] = cur;
                    if (cnt >= (size_t)B) break;
                    // read the adjacency directly: this must not depend on the
                    // materialised layer-0 CSR (which is expensive at large N)
                    u32 dc = g.deg(cur, 0);
                    const u32* nb = g.nbrs(cur, 0);
                    for (u32 k = 0; k < dc; k++) {
                        u32 v = nb[k];
                        if (!local_seen[v]) {
                            local_seen[v] = 1; touched.push_back(v);
                            q.push_back(v);
                        }
                    }
                }
                // pad deterministically if the component is smaller than B
                if (cnt < (size_t)B) {
                    u32 p = cursor;
                    for (size_t tries = 0; cnt < (size_t)B && tries < N; tries++) {
                        if (p >= N) p = 0;
                        if (!local_seen[p]) { local_seen[p] = 1; touched.push_back(p); mem[cnt++] = p; }
                        p++;
                    }
                    // every node already covered -> duplicate-pad, still in range
                    while (cnt < (size_t)B) {
                        if (p >= N) p = 0;
                        mem[cnt++] = p++;
                    }
                    cursor = (p >= N) ? 0 : p;
                }
                for (u32 v : touched) local_seen[v] = 0;
                off[gi + 1] = off[gi] + B;
            }
        }

        // coverage / replication statistics
        for (size_t gi = 0; gi < G; gi++)
            for (int j = 0; j < B; j++) stamp[members[gi * (size_t)B + j]] = 1;
        covered_cnt = 0;
        for (size_t i = 0; i < N; i++) if (stamp[i]) covered_cnt++;

        // ---- optional coverage-completion pass -------------------------------
        size_t completed = 0, spilled = 0;
        if (complete_) {
            std::vector<u32> occ(N, 0);
            for (size_t k = 0; k < members.size(); k++) occ[members[k]]++;
            size_t scan = 0;
            for (u32 v = 0; v < N; v++) {
                if (occ[v]) continue;
                while (scan < members.size() && occ[members[scan]] <= 1) scan++;
                if (scan >= members.size()) { spilled++; continue; }   // no duplicate slot left
                occ[members[scan]]--;
                members[scan] = v;
                occ[v] = 1;
                completed++;
            }
            if (completed) {
                std::fill(stamp.begin(), stamp.end(), 0);
                for (size_t gi = 0; gi < G; gi++)
                    for (int j = 0; j < B; j++) stamp[members[gi * (size_t)B + j]] = 1;
                covered_cnt = 0;
                for (size_t i = 0; i < N; i++) if (stamp[i]) covered_cnt++;
            }
        }
        if (set_aside) *set_aside = (u32)spilled;

        covered = covered_cnt;
        coverage = (double)covered_cnt / N;
        replication = (double)(G * (size_t)B) / N;
        build_sec = t.sec();
        if (verbose)
            printf("  [HGI] anchors(level>=%d)=%zu  B=%d  groups=%zu  slots=%.1fM  "
                   "coverage=%.4f  replication=%.2fx%s  (%.1fs)\n",
                   anchor_level, G, B, G, (double)G * B / 1e6, coverage, replication,
                   complete_ ? (completed ? "  [+completed]" : "  [complete: no-op]") : "",
                   build_sec);
    }
};

// ------------------------------------------------------------------ multi-index
// Per group: m independent sub-tables.  Entry = (key32 << 32) | vector_id,
// sorted by key so a lookup is a binary search plus a short run scan.
// key32 = low 32 bits of AES-PRF(segment code): deterministic encryption of keys.
struct GroupMultiIndex {
    int m = 8, bps = 16;
    int kper = 1;            // u32 words per sub-segment = ceil(bps/32)
    u32 B = 0;
    size_t G = 0;
    std::vector<u64> off;    // G*m + 1
    std::vector<u64> ent;    // sorted packed entries
    std::vector<u32> keymin, keymax;   // per (g,l): key range -> O(1) miss rejection
    double build_sec = 0.0;
    size_t unique_keys = 0;
    double avg_bucket = 0.0;

    // PRF over a kper-word segment.  Chained (CBC-MAC style) so that any change in
    // any word changes the key, and so that kper == 1 reduces exactly to the
    // historical single-word scheme  key = aes_prf(pk, seg).
    static inline u32 segkey(const AesKey& pk, const u32* seg, int kper) {
        u64 h = 0;
        for (int w = 0; w < kper; w++) h = aes_prf_u64(pk, (u64)seg[w] ^ h);
        return (u32)h;
    }

    // precomputed per-vector per-segment PRF keys (shared across groups)
    void build(const HGI& hgi, const std::vector<u32>& codes, int m_, int bps_, int kper_,
               const AesKey& pk, bool verbose = true) {
        Timer t;
        m = m_;
        bps = bps_;
        kper = (kper_ > 0) ? kper_ : (bps_ + 31) / 32;
        G = hgi.G();
        B = (u32)hgi.B;
        // Build PRF keys for every (vector, segment)
        size_t maxid = 0;
        for (u64 x : hgi.members) if (x > maxid) maxid = x;
        std::vector<u32> pkeys(((size_t)maxid + 1) * m);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i <= (long long)maxid; i++)
            for (int l = 0; l < m; l++)
                pkeys[(size_t)i * m + l] =
                    segkey(pk, codes.data() + ((size_t)i * m + l) * kper, kper);

        off.assign(G * m + 1, 0);
        ent.resize(G * (size_t)B * m);
        keymin.assign(G * m, 0xFFFFFFFFu);
        keymax.assign(G * m, 0u);

        #pragma omp parallel for schedule(dynamic, 4)
        for (long long gi = 0; gi < (long long)G; gi++) {
            const u32* mem = hgi.members.data() + (size_t)gi * B;
            for (int l = 0; l < m; l++) {
                u64 base = (u64)gi * m + l;
                u64* dst = ent.data() + base * B;
                for (u32 j = 0; j < B; j++) {
                    u32 id = mem[j];
                    dst[j] = ((u64)pkeys[(size_t)id * m + l] << 32) | (u64)id;
                }
                std::sort(dst, dst + B);
                keymin[base] = (u32)(dst[0] >> 32);
                keymax[base] = (u32)(dst[B - 1] >> 32);
            }
        }
        // prefix offsets (B entries per group-segment)
        for (size_t i = 0; i <= (size_t)G * m; i++) off[i] = (u64)i * B;

        // statistics
        size_t n_entries = (size_t)G * B * m;
        u64 sum_runs = 0;
        size_t runs = 0;
        for (size_t i = 0; i < (size_t)G * m; i += 1) {
            // sample only to keep the stat cheap
            if ((i % 97) != 0) continue;
            u64 s = off[i], e = off[i + 1];
            u32 prev = 0xFFFFFFFFu; size_t cnt = 0;
            for (u64 k = s; k < e; k++) {
                u32 key = (u32)(ent[k] >> 32);
                if (key != prev) { if (prev != 0xFFFFFFFFu) { sum_runs += cnt; runs++; } prev = key; cnt = 1; }
                else cnt++;
            }
            if (prev != 0xFFFFFFFFu) { sum_runs += cnt; runs++; }
        }
        unique_keys = runs;
        avg_bucket = runs ? (double)sum_runs / runs : 0.0;
        (void)n_entries;
        build_sec = t.sec();
        if (verbose)
            printf("  [MultiIndex] m=%d bps=%d kper=%d  groups=%zu  entries=%.1fM  mem=%.2f GB  "
                   "avg_bucket=%.3f  (%.1fs)\n",
                   m, bps, kper, G, n_entries / 1e6, (double)ent.size() * 8 / 1e9, avg_bucket, build_sec);
    }

    inline void gather(long long gi, int l, u32 key, std::vector<u32>& out) const {
        const size_t blk = (size_t)gi * m + l;
        if (key < keymin[blk] || key > keymax[blk]) return;   // O(1) reject before the search
        u64 s = off[blk], e = off[blk + 1];
        u64 target = (u64)key << 32;
        const u64* base = ent.data();
        const u64* lo = std::lower_bound(base + s, base + e, target);
        for (const u64* p = lo; p < base + e && (*p >> 32) == key; ++p)
            out.push_back((u32)(*p & 0xFFFFFFFFu));
    }
    inline void gather_sorted(long long gi, int l, u32 key, std::vector<u32>& out) const {
        out.clear(); gather(gi, l, key, out); std::sort(out.begin(), out.end());
    }

    // ---- semantics A: UNION (what the reference Python prototype implements;
    //      a candidate is kept if ANY sub-segment matches, optionally within a
    //      per-segment Hamming radius r).
    //      `td` holds the already-PRF'd trapdoors: segment l owns
    //      td[l*(nflip+1) .. l*(nflip+1)+nflip] where nflip = 0 (r=0) or bps (r=1).
    inline void filter_union_keys(long long gi, const u32* td, int nflip,
                                  std::vector<u32>& out) const {
        const int per = nflip + 1;
        for (int l = 0; l < m; l++)
            for (int j = 0; j < per; j++)
                gather(gi, l, td[l * per + j], out);
    }

    // client-side trapdoor generation: PRF over the query sub-segment and its
    // bounded bit-flip variants.
    //   kper == 1 -> identical to the historical scheme (the flip set is the same,
    //   only the order of the flips is reversed, which does not change the union).
    static void make_trapdoors(const std::vector<u32>& qcode, int m_, int bps_, int r,
                               const AesKey& pk, std::vector<u32>& td, int& nflip,
                               int kper_ = 1) {
        const int kper = (kper_ > 0) ? kper_ : 1;
        nflip = (r >= 1) ? bps_ : 0;
        const int per = nflip + 1;
        td.assign((size_t)m_ * per, 0);
        std::vector<u32> seg((size_t)kper);
        for (int l = 0; l < m_; l++) {
            const u32* q = qcode.data() + (size_t)l * kper;
            td[(size_t)l * per + 0] = segkey(pk, q, kper);
            if (r >= 1)
                for (int b = 0; b < bps_; b++) {
                    for (int w = 0; w < kper; w++) seg[w] = q[w];
                    const int pos = bps_ - 1 - b;          // same bit as the encoder
                    seg[pos >> 5] ^= 1u << (pos & 31);
                    td[(size_t)l * per + 1 + b] = segkey(pk, seg.data(), kper);
                }
        }
    }

    // ---- semantics B: PIGEONHOLE (the strict formulation in the paper text):
    //      R_r = union over flip patterns of (intersection of all sub-segments).
    //      r=0 -> exact match in every sub-segment; r=1 -> exactly one bit flip
    //      in exactly one sub-segment while all others match exactly.
    //      `td` is the PRF trapdoor bundle produced by make_trapdoors:
    //      segment l owns td[l*per .. l*per+per) with per = nflip + 1, entry 0
    //      being the exact key and entries 1..bps the single-bit-flip keys.
    //      (Never XOR the PRF value itself -- the table is keyed by PRF outputs.)
    inline void filter_pigeonhole(long long gi, const u32* td, int nflip,
                                  std::vector<u32>& out) const {
        const int per = nflip + 1;
        std::vector<std::vector<u32>> sets(m);
        for (int l = 0; l < m; l++) gather_sorted(gi, l, td[(size_t)l * per], sets[l]);

        auto inter_with_sets = [&](const std::vector<u32>& a,
                                   const std::vector<u32>& b) -> std::vector<u32> {
            std::vector<u32> r_; std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                                                       std::back_inserter(r_));
            return r_;
        };
        // R0 = intersection over all exact sub-segment sets
        if (sets[0].empty()) return;
        std::vector<u32> acc = sets[0];
        for (int l = 1; l < m && !acc.empty(); l++) acc = inter_with_sets(acc, sets[l]);
        out.insert(out.end(), acc.begin(), acc.end());
        if (nflip < 1) return;
        for (int i = 0; i < m; i++) {
            // intersection of the exact sets of every segment except i
            std::vector<u32> others = sets[(i + 1) % m];
            for (int l = 0; l < m; l++) {
                if (l == i || l == (i + 1) % m) continue;
                if (others.empty()) break;
                others = inter_with_sets(others, sets[l]);
            }
            if (others.empty()) continue;
            for (int b = 0; b < nflip; b++) {
                std::vector<u32> flip;
                gather_sorted(gi, i, td[(size_t)i * per + 1 + b], flip);
                if (flip.empty()) continue;
                std::vector<u32> r_ = inter_with_sets(flip, others);
                out.insert(out.end(), r_.begin(), r_.end());
            }
        }
    }
};

// ------------------------------------------------------------------ reverse map
// vector -> the set of groups that contain it.  Used only by the completeness
// diagnostic (E1): it lets us attribute recall loss to
//   (a) the vector never being packed into any group      -> coverage loss
//   (b) its groups not being among the tau routed groups  -> routing loss
//   (c) it being filtered out inside a routed group       -> filtering loss
struct ReverseIndex {
    std::vector<u64> off;    // N+1
    std::vector<u32> ents;   // G*B
    void build(const HGI& h, size_t N) {
        std::vector<u64> cnt(N + 1, 0);
        for (u32 v : h.members) cnt[v + 1]++;
        off.assign(N + 1, 0);
        for (size_t i = 0; i < N; i++) off[i + 1] = off[i] + cnt[i + 1];
        ents.resize(off[N]);
        std::vector<u64> cur(off.begin(), off.end() - 1);
        for (size_t g = 0; g < h.anchor.size(); g++)
            for (u32 j = 0; j < h.B; j++) ents[cur[h.members[g * (size_t)h.B + j]]++] = (u32)g;
    }
    inline size_t deg(u32 v) const { return (size_t)(off[v + 1] - off[v]); }
    inline const u32* groups(u32 v) const { return ents.data() + off[v]; }
    size_t mem_bytes() const { return off.size() * 8 + ents.size() * 4; }
};

} // namespace eh
