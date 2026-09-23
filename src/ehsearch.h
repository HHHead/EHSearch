// ehsearch.h - EHSearch client / server pipeline (in-memory backend)
#pragma once
#include "common.h"
#include "hnsw.h"
#include "rabitq.h"
#include "hgi.h"
#include "crypto.h"
#include "ssd.h"
#include <omp.h>
#include <queue>

namespace eh {

// ============================================================ client: upper-layer graph
struct UpperGraph {
    int d = 128;
    int max_level = 0;
    u32 entry = 0;
    std::vector<u32> nodes;        // global ids with hnsw level >= 1
    std::vector<i32> node_idx;     // global id -> local index
    std::vector<float> vecs;       // local x d
    std::vector<i32> lvl;
    std::vector<std::vector<u64>> loff;
    std::vector<std::vector<u32>> lnbr;

    size_t mem_bytes() const {
        size_t b = vecs.size() * 4 + node_idx.size() * 4 + nodes.size() * 4 + lvl.size() * 4;
        for (auto& v : lnbr) b += v.size() * 4;
        for (auto& v : loff) b += v.size() * 8;
        return b;
    }

    void build(const HNSW& g) {
        d = g.d;
        node_idx.assign(g.n, -1);
        nodes.clear();
        for (u32 i = 0; i < g.n; i++)
            if (g.levels[i] >= 1) { node_idx[i] = (i32)nodes.size(); nodes.push_back(i); }
        const size_t L = nodes.size();
        vecs.resize(L * d);
        lvl.resize(L);
        for (size_t j = 0; j < L; j++) {
            memcpy(&vecs[j * d], g.vec(nodes[j]), (size_t)d * 4);
            lvl[j] = g.levels[nodes[j]];
        }
        max_level = g.max_level;
        entry = g.entry_point;
        loff.assign(max_level + 1, {});
        lnbr.assign(max_level + 1, {});
        for (int l = 1; l <= max_level; l++) {
            loff[l].assign(L + 1, 0);
            for (size_t j = 0; j < L; j++) {
                u32 nd = nodes[j];
                u32 c = (lvl[j] >= l) ? g.deg(nd, l) : 0;
                loff[l][j + 1] = loff[l][j] + c;
            }
            lnbr[l].resize(loff[l][L]);
            for (size_t j = 0; j < L; j++) {
                if (lvl[j] < l) continue;
                u32 nd = nodes[j], c = g.deg(nd, l);
                const u32* nb = g.nbrs(nd, l);
                for (u32 k = 0; k < c; k++) lnbr[l][loff[l][j] + k] = (u32)node_idx[nb[k]];
            }
        }
    }

    inline const float* v(i32 local) const { return vecs.data() + (size_t)local * d; }

    // greedy descent through layers >= 2, then best-first exploration of layer 1
    void route(const float* q, int tau, int ef, std::vector<u32>& out) const {
        out.clear();
        if (nodes.empty()) return;
        i32 cur = node_idx[entry];
        if (cur < 0) cur = 0;
        float curd = l2_sqr(q, v(cur), d);
        for (int l = max_level; l >= 2; l--) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (u64 k = loff[l][cur]; k < loff[l][cur + 1]; k++) {
                    i32 nb = (i32)lnbr[l][k];
                    float dd = l2_sqr(q, v(nb), d);
                    if (dd < curd) { curd = dd; cur = nb; changed = true; }
                }
            }
        }
        if (max_level < 1) { out.push_back((u32)cur); return; }
        const int l = 1;
        const size_t L = nodes.size();
        thread_local std::vector<u32> stamp;
        thread_local u32 epoch = 0;
        if (stamp.size() < L) stamp.resize(L, 0);
        epoch++; if (epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0); epoch = 1; }
        using CD = std::pair<float, i32>;
        std::priority_queue<CD, std::vector<CD>, std::greater<CD>> cand;
        std::priority_queue<CD> res;
        cand.push({curd, cur}); res.push({curd, cur}); stamp[cur] = epoch;
        while (!cand.empty()) {
            auto c = cand.top(); cand.pop();
            if ((int)res.size() >= ef && c.first > res.top().first) break;
            for (u64 k = loff[l][c.second]; k < loff[l][c.second + 1]; k++) {
                i32 nb = (i32)lnbr[l][k];
                if (stamp[nb] == epoch) continue;
                stamp[nb] = epoch;
                float dd = l2_sqr(q, v(nb), d);
                if ((int)res.size() < ef || dd < res.top().first) {
                    cand.push({dd, nb}); res.push({dd, nb});
                    if ((int)res.size() > ef) res.pop();
                }
            }
        }
        std::vector<std::pair<float, i32>> all;
        while (!res.empty()) { all.push_back(res.top()); res.pop(); }
        std::sort(all.begin(), all.end());
        for (size_t i = 0; i < all.size() && (int)out.size() < tau; i++) out.push_back((u32)all[i].second);
    }
};

// ============================================================ server
struct Server {
    const GroupMultiIndex* mi = nullptr;
    u32 stride = 0;
    u32 d = 0;
    PayloadStore store;      // MEM (resident) or SSD (mmap) encrypted payload backend

    size_t payload_bytes() const { return store.bytes(); }
    bool ssd_mode() const { return store.ssd; }
    double mem_mb() const {
        double b = (double)store.mem_mb();
        if (mi) b += (double)mi->ent.size() * 8 + (double)mi->off.size() * 8 + (double)mi->keymin.size() * 8;
        return b / 1048576.0;
    }

    void build_payload(const float* xb, size_t n, int d_, const AesKey& k, u64 seed = 12345) {
        d = d_;
        const u32 body = (u32)(((size_t)d * 4 + 15) / 16 * 16);
        stride = body + 16;
        store.ssd = false; store.stride = stride; store.d = (u32)d;
        store.mem.assign((size_t)n * stride, 0);
        #pragma omp parallel
        {
            std::vector<u8> tmp(body);
            std::mt19937_64 r2(seed + omp_get_thread_num());
            #pragma omp for schedule(static)
            for (long long i = 0; i < (long long)n; i++) {
                u8* rec = store.mem.data() + (size_t)i * stride;
                for (int j = 0; j < 16; j++) rec[j] = (u8)(r2() & 0xFF);
                memcpy(tmp.data(), xb + (size_t)i * d, (size_t)d * 4);
                aes_cbc_encrypt(k, rec, tmp.data(), body / 16, rec + 16);
            }
        }
    }

    // SSD / out-of-core backend: mmap an already-prepared ciphertext file
    bool attach_payload_ssd(const std::string& path, u32 d_) {
        d = d_;
        const u32 body = (u32)(((size_t)d * 4 + 15) / 16 * 16);
        stride = body + 16;
        return store.open_ssd(path, stride, (u32)d);
    }

    // exact encrypted lookups inside the target groups, then dedup
    // `inner_threads > 1` parallelises the per-group lookups inside a single
    // query.  This only helps end-to-end *latency* (single-query runs); for
    // throughput runs the queries themselves are already parallel, so the
    // caller must pass 0 there to avoid oversubscription.
    void filter(const u32* td, int nflip, const std::vector<i32>& groups, int mode,
                std::vector<u32>& cand, std::vector<u32>& seen, u32& epoch,
                int inner_threads = 0) const {
        cand.clear();
        epoch++;
        if (epoch == 0) { std::fill(seen.begin(), seen.end(), 0); epoch = 1; }
        const int ngrp = (int)groups.size();
        if (inner_threads > 1 && ngrp >= 4) {
            // per-group buckets, then merge with dedup (order of `cand` differs
            // from the serial path but the candidate SET is identical)
            // NOTE: must NOT be thread_local -- a thread_local vector is empty in
            // the worker threads, so indexing it inside the parallel region would
            // read out of bounds.  A plain local vector is safe because each
            // thread writes a distinct element.
            std::vector<std::vector<u32>> buckets((size_t)ngrp);
            #pragma omp parallel for schedule(dynamic, 1) num_threads(inner_threads)
            for (int t = 0; t < ngrp; t++) {
                if (groups[t] < 0) continue;
                if (mode == 0) mi->filter_union_keys(groups[t], td, nflip, buckets[t]);
                else          mi->filter_pigeonhole(groups[t], td, nflip, buckets[t]);
            }
            for (int t = 0; t < ngrp; t++)
                for (u32 id : buckets[t])
                    if (seen[id] != epoch) { seen[id] = epoch; cand.push_back(id); }
            return;
        }
        std::vector<u32> tmp;
        for (i32 g : groups) {
            if (g < 0) continue;
            tmp.clear();
            if (mode == 0) mi->filter_union_keys(g, td, nflip, tmp);
            else          mi->filter_pigeonhole(g, td, nflip, tmp);
            for (u32 id : tmp)
                if (seen[id] != epoch) { seen[id] = epoch; cand.push_back(id); }
        }
    }

    void fetch(const std::vector<u32>& cand, std::vector<u8>& buf, IoStats* st = nullptr) const {
        store.fetch(cand, buf, st);
    }
};

// ============================================================ per-query scratch
struct Scratch {
    std::vector<u32> groups;
    std::vector<i32> gidx;
    std::vector<u32> qcode;
    std::vector<u32> td;
    std::vector<u32> cand;
    std::vector<u32> seen;
    u32 epoch = 0;
    std::vector<u8> resp;
    std::vector<float> dv;
    std::vector<float> dist;
    std::vector<u32> ord;            // decryption order (ascending candidate id)
    std::vector<float> rot, xn;      // per-query scratch for serial RaBitQ encoding
    std::vector<u32> topk;
    void init(size_t N, int d, int m, int kper = 1) {
        seen.assign(N, 0);
        rot.assign(d, 0.f);
        xn.assign(d, 0.f);
        qcode.assign((size_t)m * std::max(1, kper), 0);
    }
};

// ============================================================ runtime optimisation switches
// All switches are runtime flags so that baseline and optimised runs can be
// produced by the SAME binary, which removes compiler-version noise from the
// A/B comparison.
struct Opts {
    bool zerocopy   = false;  // decrypt payload records in place instead of copying |C|*stride bytes
    bool omp_groups = false;  // parallelise the per-group encrypted lookups (server-side)
    bool prefetch   = false;  // explicit prefetch of the candidate vectors used by reranking
    int  inner_threads = 0;   // >1 enables intra-query parallelism (latency runs only)
};

struct QueryMetrics {
    double route_ms = 0, trapgen_ms = 0, server_ms = 0, transfer_ms = 0, decrypt_ms = 0, rerank_ms = 0;
    double compute_ms = 0;
    u64 req_bytes = 0, resp_bytes = 0;
    size_t n_groups = 0, n_cand = 0;
    IoStats io;                       // SSD backend I/O accounting
    std::vector<u32> result;
    // populated only when the completeness diagnostic (E1) is enabled
    std::vector<u32> cand_diag;
    std::vector<i32> gidx_diag;
};

// ============================================================ the system
struct EHSearch {
    // configuration
    int d = 128, m = 8, bps = 16, kper = 1, r = 0, k = 20;
    int filter_mode = 0;         // 0 = union (paper's prototype), 1 = pigeonhole (paper text)
    int ef_route = 32;
    bool do_crypto = true;       // perform real AES-NI work
    u8 prf_seed[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    u8 payload_seed[16] = {21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};

    AesKey prf_key, payload_key;

    Opts opts;

    // state
    RaBitQ enc;
    HGI hgi;
    GroupMultiIndex mi;
    UpperGraph ug;
    Server srv;
    std::vector<i32> node2group;   // client-held anchor->group map
    std::vector<u32> codes;        // N x m packed segment codes (build-time)

    // sizes
    double client_mem_mb() const {
        double b = (double)ug.mem_bytes() + (double)node2group.size() * 4 + (double)enc.W.size() * 4 * 2;
        return b / 1048576.0;
    }
    double server_mem_mb() const { return srv.mem_mb() + (double)hgi.anchor.size() * 4 / 1048576.0; }

    void init_crypto() { aes_expand(prf_seed, prf_key); aes_expand(payload_seed, payload_key); }

    // build index structures (everything except the payload store, which is
    // independent of m / B and therefore built once per process)
    // bps_ <= 0 means "auto" = max(1, d/m), the historical behaviour.
    void build_index(const HNSW& g, const float* xb, size_t n, int B, int anchor_level,
                     int m_, int bps_ = 0, bool verbose = true, bool hgi_complete = false) {
        Timer t;
        d = g.d; m = m_;
        bps = (bps_ > 0) ? bps_ : std::max(1, d / m);
        kper = (bps + 31) / 32;
        init_crypto();
        printf("  [Build] RaBitQ d=%d m=%d bps=%d kper=%d used=%d ...\n",
               d, m, bps, kper, std::min(d, m * bps));
        enc.init(d, m, bps, 42);
        std::vector<u32> codes;
        enc.encode(xb, n, codes);
        printf("  [Build] HGI ...\n");
        hgi.build(g, B, anchor_level, verbose, hgi_complete);
        node2group.assign(n, -1);
        for (size_t gi = 0; gi < hgi.anchor.size(); gi++) node2group[hgi.anchor[gi]] = (i32)gi;
        printf("  [Build] Group-Scoped Multi-Index ...\n");
        mi.build(hgi, codes, m, bps, enc.kper, prf_key, verbose);
        std::vector<u32>().swap(codes);
        printf("  [Build] client upper-layer graph ...\n");
        ug.build(g);
        srv.mi = &mi;
        if (verbose)
            printf("  [Build] index done in %.1fs | client %.1f MB | server index %.1f MB\n",
                   t.sec(), client_mem_mb(), (double)mi.ent.size() * 8 / 1048576.0);
    }

    void build_payload(const float* xb, size_t n) {
        init_crypto();
        srv.build_payload(xb, n, d, payload_key);
        printf("  [Build] encrypted payload: %.3f GB (stride %u B/vector)\n",
               (double)srv.payload_bytes() / 1e9, srv.stride);
    }

    // SSD / out-of-core backend: mmap an already prepared ciphertext file
    bool attach_payload_ssd(const std::string& path, u32 d_) {
        init_crypto();
        if (!srv.attach_payload_ssd(path, d_)) {
            fprintf(stderr, "[SSD] failed to map %s\n", path.c_str());
            return false;
        }
        // sanity: the file must hold at least the whole database
        return true;
    }

    // full query with an explicit target count.
    // `capture_diag` stores the candidate set and the routed group ids so that the
    // completeness diagnostic can attribute recall loss; it is off by default
    // because it perturbs the measured latency.
    void query(const float* q, int tau, Scratch& sc, QueryMetrics& qm,
               const LinkSim* link, bool capture_diag = false) const {
        qm = QueryMetrics();
        Timer t;
        // ---- 1) LocRout
        ug.route(q, tau, std::max(ef_route, tau), sc.groups);   // local indices
        sc.gidx.clear();
        for (u32 li : sc.groups) {
            i32 g = node2group[ug.nodes[li]];
            if (g >= 0) sc.gidx.push_back((u32)g);
        }
        qm.route_ms = t.ms();
        if (capture_diag) qm.gidx_diag = sc.gidx;
        qm.n_groups = sc.gidx.size();

        // ---- 2) TrapGen
        t.reset();
        enc.encode_one_into(q, sc.xn.data(), sc.rot.data(), sc.qcode.data());
        int nflip = 0;
        if (do_crypto) GroupMultiIndex::make_trapdoors(sc.qcode, m, bps, r, prf_key, sc.td, nflip, kper);
        else {
            // no-crypto control path (debug only): the trapdoors are the raw segment
            // values.  kper == 1 keeps the historical expression verbatim; kper > 1
            // folds the words into one 32-bit key.
            nflip = (r >= 1) ? bps : 0;
            const int per = nflip + 1;
            sc.td.assign((size_t)m * per, 0);
            for (int l = 0; l < m; l++) {
                const u32* qc = sc.qcode.data() + (size_t)l * kper;
                if (kper == 1) {
                    sc.td[(size_t)l * per + 0] = qc[0];
                    if (r >= 1)
                        for (int b = 0; b < bps; b++)
                            sc.td[(size_t)l * per + 1 + b] = qc[0] ^ (1u << b);
                } else {
                    auto fold = [&](const u32* w) {
                        u32 h = 2166136261u;
                        for (int t = 0; t < kper; t++) { h ^= w[t]; h *= 16777619u; }
                        return h;
                    };
                    sc.td[(size_t)l * per + 0] = fold(qc);
                    if (r >= 1) {
                        std::vector<u32> seg((size_t)kper);
                        for (int b = 0; b < bps; b++) {
                            for (int w = 0; w < kper; w++) seg[w] = qc[w];
                            const int pos = bps - 1 - b;
                            seg[pos >> 5] ^= 1u << (pos & 31);
                            sc.td[(size_t)l * per + 1 + b] = fold(seg.data());
                        }
                    }
                }
            }
        }
        qm.trapgen_ms = t.ms();

        qm.req_bytes = 16 + (u64)sc.gidx.size() * 4 + (u64)m * (nflip + 1) * 4;

        // ---- 3) Filter (server, exact encrypted lookups)
        t.reset();
        srv.filter(sc.td.data(), nflip, sc.gidx, filter_mode, sc.cand, sc.seen, sc.epoch,
                   opts.omp_groups ? opts.inner_threads : 0);
        qm.n_cand = sc.cand.size();
        if (capture_diag) qm.cand_diag = sc.cand;
        const u32 stride = srv.stride;
        if (!opts.zerocopy) {
            srv.fetch(sc.cand, sc.resp, &qm.io);
        } else {
            // zero-copy: the client decrypts straight out of the payload
            qm.io.bytes_fetched += (u64)sc.cand.size() * stride;
        }
        qm.server_ms = t.ms();
        qm.resp_bytes = 16 + (u64)sc.cand.size() * 4 + (u64)sc.cand.size() * stride;

        // ---- 4) network
        qm.transfer_ms = link ? link->added_sec(qm.req_bytes, qm.resp_bytes) * 1e3 : 0.0;

        // ---- 5) Refine (client: decrypt + exact L2 + top-k)
        t.reset();
        const u32 body = stride - 16;
        const size_t nc = sc.cand.size();
        sc.dv.resize(nc * (size_t)d);

        // Payload source.  In the zero-copy path the ciphertext records are read
        // in place; both backends store records in ascending id order, so
        // decrypting in ascending-id order keeps the access pattern monotone
        // (sequential SSD reads / sequential DRAM sweeps).
        const u8* src = nullptr;
        if (opts.zerocopy) src = srv.store.ssd ? srv.store.map.base() : srv.store.mem.data();

        std::vector<u32>& ord = sc.ord;
        ord.resize(nc);
        for (u32 i = 0; i < (u32)nc; i++) ord[i] = i;
        if (opts.zerocopy)
            std::sort(ord.begin(), ord.end(),
                      [&](u32 a2, u32 b2) { return sc.cand[a2] < sc.cand[b2]; });

        for (size_t z = 0; z < nc; z++) {
            const u32 i = ord[z];
            const u8* rec;
            if (src) rec = src + (size_t)sc.cand[i] * stride;
            else     rec = sc.resp.data() + (size_t)i * stride;
            if (opts.prefetch && z + 6 < nc) {
                const u32 nxt = ord[z + 6];
                const u8* base = src ? src : sc.resp.data();
                __builtin_prefetch(base + (size_t)sc.cand[nxt] * stride, 0, 1);
            }
            if (do_crypto)
                aes_cbc_decrypt(payload_key, rec, rec + 16, body / 16,
                                (u8*)(sc.dv.data() + (size_t)i * d));
            else
                memcpy(sc.dv.data() + (size_t)i * d, rec + 16, (size_t)d * 4);
        }
        qm.decrypt_ms = t.ms();

        t.reset();
        sc.dist.resize(nc);
        for (size_t i = 0; i < nc; i++) {
            if (opts.prefetch && i + 8 < nc)
                __builtin_prefetch(sc.dv.data() + (size_t)(i + 8) * d, 0, 0);
            sc.dist[i] = l2_sqr(q, sc.dv.data() + (size_t)i * d, d);
        }
        topk_smallest(sc.dist.data(), sc.dist.size(), (size_t)k, sc.topk);
        qm.result.clear();
        for (u32 j : sc.topk) qm.result.push_back(sc.cand[j]);
        qm.rerank_ms = t.ms();

        qm.compute_ms = qm.route_ms + qm.trapgen_ms + qm.server_ms + qm.decrypt_ms + qm.rerank_ms;
    }

    // recall of a metrics result vs ground truth.
    // Only the first min(k, gt_k) returned results are allowed to contribute.
    static double recall(const QueryMetrics& qm, const float*  /*unused*/, const i32* gt_row, int gt_k, int k) {
        if (qm.result.empty()) return 0.0;
        const int kk = std::min(k, gt_k);
        const int nres = std::min<int>(kk, (int)qm.result.size());
        int h = 0;
        for (int a = 0; a < nres; a++) {
            for (int j = 0; j < kk; j++) if ((i32)qm.result[a] == gt_row[j]) { h++; break; }
        }
        return (double)h / k;
    }
};

} // namespace eh
