// main.cpp - EHSearch C++ benchmark driver
//   ehsearch hnsw  --data DIR --cache FILE [--M 32 --efC 64]
//   ehsearch bench --data DIR --cache FILE --out FILE [options]
#include "common.h"
#include "simd.h"
#include "hnsw.h"
#include "rabitq.h"
#include "hgi.h"
#include "crypto.h"
#include "ehsearch.h"
#include <omp.h>

using namespace eh;

static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out; std::string cur;
    for (char ch : s) { if (ch == c) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += ch; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
static std::vector<int> split_int(const std::string& s, char c = ',') {
    std::vector<int> out; for (auto& t : split(s, c)) out.push_back(atoi(t.c_str())); return out;
}
// CSV-safe rendering of the optimisation tag (optstr contains commas)
static std::string plus_join(const std::string& s) {
    std::string o = s;
    for (auto& c : o) if (c == ',') c = '+';
    return o;
}

static std::vector<double> split_dbl(const std::string& s, char c = ',') {
    std::vector<double> out; for (auto& t : split(s, c)) out.push_back(atof(t.c_str())); return out;
}
static std::string arg(int argc, char** argv, const std::string& k, const std::string& def) {
    for (int i = 1; i + 1 < argc; i++) if (k == argv[i]) return argv[i + 1];
    return def;
}
static bool flag(int argc, char** argv, const std::string& k) {
    for (int i = 1; i < argc; i++) if (k == argv[i]) return true;
    return false;
}

struct Dataset {
    std::vector<float> xb, xq;
    std::vector<i32> gt;
    size_t nb = 0, nq = 0; int d = 0, dgt = 0;
};
// ------------------------------------------------------------------ dataset loading
// Default: the irisa SIFT-1M bundle.  For the out-of-core / scale experiments the
// bigann (SIFT-1B family) files are used instead:
//   --base-raw <sift1b_raw_uint8.bin>  --nbase N
//   --query    <bigann_query.bvecs>    --gt <gnd/idx_<N>.ivecs>
static std::vector<float> read_any_query(const std::string& path, size_t* n, int* d) {
    if (path.size() > 6 && path.compare(path.size() - 6, 6, ".bvecs") == 0)
        return read_bvecs_as_float(path, n, d);
    return read_fvecs(path, n, d);
}
static Dataset load_dataset(const std::string& dir, const std::string& base_raw, size_t nbase,
                            const std::string& qpath, const std::string& gpath,
                            const std::string& bpath, int raw_dim = 128) {
    Dataset ds;
    if (!base_raw.empty()) {
        int d0 = raw_dim;
        ds.xb = read_raw_u8_as_float(base_raw, nbase, d0);
        ds.nb = nbase; ds.d = d0;
    } else if (!bpath.empty()) {
        ds.xb = read_fvecs(bpath, &ds.nb, &ds.d);
    } else {
        ds.xb = read_fvecs(dir + "/sift_base.fvecs", &ds.nb, &ds.d);
    }
    if (!qpath.empty()) ds.xq = read_any_query(qpath, &ds.nq, &ds.d);
    else                ds.xq = read_fvecs(dir + "/sift_query.fvecs", &ds.nq, &ds.d);
    size_t ng; int dg;
    if (!gpath.empty()) ds.gt = read_ivecs(gpath, &ng, &dg);
    else                ds.gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &ng, &dg);
    ds.dgt = dg;
    printf("[Data] base %zu x %d | query %zu x %d | gt %zu x %d\n",
           ds.nb, ds.d, ds.nq, ds.d, ng, dg);
    return ds;
}

// ------------------------------------------------------------------ client / server disk artefacts
struct DiskFootprint { u64 client = 0, server = 0; };
static DiskFootprint measure_disk(const EHSearch& sys, bool write_files, const std::string& outdir) {
    DiskFootprint fp;
    // client: upper-layer vectors + adjacency + anchor->group map + rotation matrix
    u64 c = (u64)sys.ug.vecs.size() * 4 + (u64)sys.node2group.size() * 4
            + (u64)sys.enc.W.size() * 4;
    for (auto& v : sys.ug.lnbr) c += (u64)v.size() * 4;
    for (auto& v : sys.ug.loff) c += (u64)v.size() * 8;
    fp.client = c;
    u64 s = (u64)sys.srv.payload_bytes()
            + (u64)sys.mi.ent.size() * 8 + (u64)sys.mi.off.size() * 8
            + (u64)sys.mi.keymin.size() * 4 * 2;
    fp.server = s;
    if (write_files) {
        std::filesystem::create_directories(outdir);
        auto dump = [](const std::string& p, const void* d, size_t b) {
            std::ofstream f(p, std::ios::binary); f.write((const char*)d, b);
        };
        dump(outdir + "/client_upper_adj.bin", sys.ug.lnbr.empty() ? nullptr : sys.ug.lnbr[1].data(),
             sys.ug.lnbr.empty() ? 0 : sys.ug.lnbr[1].size() * 4);
        dump(outdir + "/client_node2group.bin", sys.node2group.data(), sys.node2group.size() * 4);
        dump(outdir + "/client_rotation.bin", sys.enc.W.data(), sys.enc.W.size() * 4);
        // write upper-layer vectors layer by layer to keep the buffer bounded
        { std::ofstream f(outdir + "/client_upper_vecs.bin", std::ios::binary);
          f.write((const char*)sys.ug.vecs.data(), (std::streamsize)sys.ug.vecs.size() * 4); }
        { std::ofstream f(outdir + "/server_multiindex.bin", std::ios::binary);
          f.write((const char*)sys.mi.ent.data(), (std::streamsize)sys.mi.ent.size() * 8);
          f.write((const char*)sys.mi.off.data(), (std::streamsize)sys.mi.off.size() * 8);
          f.write((const char*)sys.mi.keymin.data(), (std::streamsize)sys.mi.keymin.size() * 4);
          f.write((const char*)sys.mi.keymax.data(), (std::streamsize)sys.mi.keymax.size() * 4); }
        if (!sys.srv.ssd_mode()) {   // the SSD backend already lives in its own file
            std::ofstream f(outdir + "/server_payload.bin", std::ios::binary);
            f.write((const char*)sys.srv.store.mem.data(),
                    (std::streamsize)sys.srv.store.mem.size());
        }
        // recount from real files
        u64 c2 = 0, s2 = 0;
        for (auto& n : {"client_upper_adj.bin", "client_node2group.bin", "client_rotation.bin",
                        "client_upper_vecs.bin"}) c2 += file_size(outdir + "/" + n);
        for (auto& n : {"server_multiindex.bin", "server_payload.bin"}) s2 += file_size(outdir + "/" + n);
        fp.client = c2; fp.server = s2;
    }
    return fp;
}

// ------------------------------------------------------------------ payload file preparation
// Streams the raw uint8 base through AES-128-CBC and writes the on-disk ciphertext
// store that the SSD backend mmaps.  Block-wise so memory stays O(block).
static void prepare_payload_file(const std::string& out_path, const std::string& base_raw,
                                 size_t n, int d, const AesKey& key, u64 seed = 12345) {
    const u32 body = (u32)(((size_t)d * 4 + 15) / 16 * 16);
    const u32 stride = body + 16;
    std::ifstream in(base_raw, std::ios::binary);
    if (!in) { fprintf(stderr, "[prepare] cannot open %s\n", base_raw.c_str()); exit(1); }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { fprintf(stderr, "[prepare] cannot write %s\n", out_path.c_str()); exit(1); }
    const size_t BLK = 65536;
    std::vector<u8> raw(BLK * (size_t)d);
    std::vector<u8> rec(stride);
    std::vector<u8> plain(body);
    std::mt19937_64 rng(seed);
    Timer t; size_t done = 0;
    while (done < n) {
        const size_t k = std::min(BLK, n - done);
        in.read((char*)raw.data(), (std::streamsize)(k * (size_t)d));
        if ((size_t)in.gcount() != k * (size_t)d) {
            fprintf(stderr, "[prepare] short read at %zu\n", done); exit(1);
        }
        for (size_t i = 0; i < k; i++) {
            for (int j = 0; j < 16; j++) rec[j] = (u8)(rng() & 0xFF);
            float* pf = (float*)plain.data();
            for (int j = 0; j < d; j++) pf[j] = (float)raw[i * (size_t)d + j];
            aes_cbc_encrypt(key, rec.data(), plain.data(), body / 16, rec.data() + 16);
            out.write((const char*)rec.data(), stride);
        }
        done += k;
        if ((done % (n / 10 + 1)) < BLK)
            printf("  [prepare] %.1f%%  (%.0fs)\n", 100.0 * done / n, t.sec());
    }
    out.flush(); out.close();
    printf("[prepare] wrote %s : %.2f GB  (n=%zu, d=%d, stride=%u B, %.0fs)\n",
           out_path.c_str(), (double)file_size(out_path) / 1e9, n, d, stride, t.sec());
}

// ------------------------------------------------------------------ benchmark
static double run_pass(EHSearch& sys, const Dataset& ds, int tau, int nq, int nthreads,
                       const LinkSim& link, const LinkSim* link2, int k_max,
                       std::vector<QueryMetrics>* out, std::vector<double>* rec,
                       std::vector<double>* rec10, std::vector<double>* rec20,
                       std::vector<double>* rec100, double* wall_sec) {
    Timer tw;
    // Intra-query parallelism only helps when queries are NOT already run in
    // parallel; enabling it in the throughput pass would oversubscribe.
    const int inner_saved = sys.opts.inner_threads;
    sys.opts.inner_threads = (nthreads == 1) ? inner_saved : 0;
    const int k1 = 1, k10 = 10, k20 = 20, k100 = 100;
    std::vector<QueryMetrics> local(nq);
    std::vector<double> r1(nq), r10(nq), r20(nq), r100(nq);
    #pragma omp parallel num_threads(nthreads)
    {
        Scratch sc; sc.init(ds.nb, ds.d, sys.m, sys.kper);
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < nq; i++) {
            sys.query(ds.xq.data() + (size_t)i * ds.d, tau, sc, local[i], &link);
            const i32* g = ds.gt.data() + (size_t)i * ds.dgt;
            auto rc = [&](int kk) {
                // Recall@kk = |returned top-kk  INTERSECT  ground-truth top-kk| / kk.
                // IMPORTANT: only the first kk returned results may be counted --
                // scanning the whole k=100 result list inflates Recall@1/@10/@20.
                const int nres = std::min<int>(kk, (int)local[i].result.size());
                const int ngt  = std::min<int>(kk, ds.dgt);
                int h = 0;
                for (int a = 0; a < nres; a++)
                    for (int j = 0; j < ngt; j++)
                        if ((i32)local[i].result[a] == g[j]) { h++; break; }
                return (double)h / kk;
            };
            r1[i] = rc(k1); r10[i] = rc(k10); r20[i] = rc(k20); r100[i] = rc(k100);
        }
    }
    sys.opts.inner_threads = inner_saved;
    *wall_sec = tw.sec();
    if (out) *out = std::move(local);
    if (rec) *rec = std::move(r1);
    if (rec10) *rec10 = std::move(r10);
    if (rec20) *rec20 = std::move(r20);
    if (rec100) *rec100 = std::move(r100);
    (void)link2; (void)k_max;
    return 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: never lose logs on a crash
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::string cmd = argc > 1 ? argv[1] : "help";
    std::string data = arg(argc, argv, "--data",
        "./data/sift");
    std::string cache = arg(argc, argv, "--cache", "./build/hnsw_M32_efC64.bin");
    std::string out   = arg(argc, argv, "--out", "./results/bench.csv");
    std::string artifact_dir = arg(argc, argv, "--artifact-dir",
        "./build/artifacts");
    int M       = atoi(arg(argc, argv, "--M", "32").c_str());
    int efC     = atoi(arg(argc, argv, "--efC", "64").c_str());
    int nthreads = atoi(arg(argc, argv, "--threads", "32").c_str());
    int anchor_level = atoi(arg(argc, argv, "--anchor-level", "1").c_str());
    int k_max   = atoi(arg(argc, argv, "--k", "100").c_str());
    int ef_route = atoi(arg(argc, argv, "--ef-route", "32").c_str());
    int nq_max  = atoi(arg(argc, argv, "--nq", "10000").c_str());
    int nq_1t   = atoi(arg(argc, argv, "--nq-1t", "500").c_str());
    auto m_list  = split_int(arg(argc, argv, "--m-list", "8"));
    auto bs_list = split_int(arg(argc, argv, "--bs-list", "1500"));
    auto r_list  = split_int(arg(argc, argv, "--r-list", "0"));
    auto tau_list = split_int(arg(argc, argv, "--tau-list", "1,5,10,20,30,40,50"));
    std::string fmodes = arg(argc, argv, "--filter", "union");
    int k_eval   = atoi(arg(argc, argv, "--k-eval", "10").c_str());
    // ---- dimension-ablation controls -------------------------------------
    // --bps N : bits per RaBitQ sub-segment.  0 (default) = auto = max(1, d/m),
    //           which reproduces the historical (d=128, m in {4,8,10,16}) grid.
    //           Setting it explicitly decouples the segment width from d, so the
    //           same bps can be compared across datasets of different dimension;
    //           the code then covers min(d, m*bps) dimensions.
    // --dim N : dimension of the RAW (headerless) base blob used by --base-raw and
    //           by `prepare`.  Ignored for .fvecs input, where the header wins.
    int bps_arg  = atoi(arg(argc, argv, "--bps", "0").c_str());
    int raw_dim  = atoi(arg(argc, argv, "--dim", "128").c_str());
    // --hgi-complete: run the coverage-completion pass (rewrite duplicated padding
    // slots with still-uncovered ids) so HGI coverage is exactly 1.0 even when the
    // layer-0 giant component is smaller than N (GIST-1M: 98.48%).
    const bool hgi_complete = flag(argc, argv, "--hgi-complete");
    // ---- optimisation switches (same binary -> clean A/B comparison)
    Opts OP;
    std::string optstr = arg(argc, argv, "--opts", "");
    {
        for (auto& t : split(optstr, ',')) {
            if (t == "zerocopy")        OP.zerocopy = true;
            else if (t == "omp_groups") OP.omp_groups = true;
            else if (t == "prefetch")   OP.prefetch = true;
            else if (t == "inner")      OP.inner_threads = nthreads;
            else if (t == "all") { OP.zerocopy = OP.omp_groups = OP.prefetch = true;
                                   OP.inner_threads = nthreads; }
        }
    }
    omp_set_max_active_levels(2);
    auto lan = split_dbl(arg(argc, argv, "--lan", "3.0,1.0"));
    auto wan = split_dbl(arg(argc, argv, "--wan", "0.4,80.0"));
    bool no_crypto = flag(argc, argv, "--no-crypto");
    bool write_art = flag(argc, argv, "--write-artifacts");
    // out-of-core / scale experiment options
    std::string backend    = arg(argc, argv, "--backend", "mem");        // mem | ssd
    std::string payload_f  = arg(argc, argv, "--payload-file", "");
    std::string base_raw   = arg(argc, argv, "--base-raw", "");
    size_t nbase           = (size_t)atoll(arg(argc, argv, "--nbase", "1000000").c_str());
    std::string qpath      = arg(argc, argv, "--query", "");
    std::string gpath      = arg(argc, argv, "--gt", "");
    std::string bpath      = arg(argc, argv, "--base", "");
    std::string tag        = arg(argc, argv, "--tag", "");

    if (cmd == "help" || cmd.empty()) {
        printf("usage: ehsearch <hnsw|prepare|bench> [options]\n"
               "  bench options: --backend mem|ssd --payload-file F --base-raw F --nbase N\n"
               "                 --query Q.bvecs --gt idx_N.ivecs --tag TAG\n"
               "                 --base BASE.fvecs        (any dimension; header wins)\n"
               "                 --m-list M --bps B --dim D   (bps=0 -> auto d/m)\n"
               "  prepare: --prepare-payload --base-raw F --nbase N --dim D --payload-file F\n");
        return 0;
    }
    print_cpu_caps();
    omp_set_num_threads(nthreads);
    printf("[CPU] threads=%d\n", nthreads);

    // ---------------- standalone payload preparation (writes the ciphertext file)
    if (cmd == "prepare" || flag(argc, argv, "--prepare-payload")) {
        if (base_raw.empty() || payload_f.empty()) {
            fprintf(stderr, "prepare: --base-raw and --payload-file are required\n");
            return 1;
        }
        EHSearch tmp; tmp.init_crypto();
        prepare_payload_file(payload_f, base_raw, nbase, raw_dim, tmp.payload_key);
        return 0;
    }

    // ---------------- coverage-only sweep -------------------------------------
    // The HGI coverage depends ONLY on the cached graph (group BFS + padding), not
    // on the vectors or the multi-index.  This path therefore skips loading the
    // base entirely, so a sweep over many B values takes seconds instead of
    // minutes -- that is what makes the "minimum B for 100% coverage" question
    // answerable without a long experiment.  Must run BEFORE load_dataset().
    if (flag(argc, argv, "--hgi-only")) {
        HNSW gg;
        if (!gg.load(cache, nullptr)) {
            fprintf(stderr, "[hgi-only] cannot load graph cache %s\n", cache.c_str());
            return 1;
        }
        const bool hc = flag(argc, argv, "--hgi-complete");
        printf("[hgi-only] graph n=%zu  sweep over B=%s  anchor_level=%d  complete=%d\n",
               gg.n, arg(argc, argv, "--bs-list", "1500").c_str(), anchor_level, (int)hc);
        for (int Bv : bs_list) {
            HGI h;
            h.build(gg, Bv, anchor_level, true, hc);
        }
        return 0;
    }

    Dataset ds = load_dataset(data, base_raw, nbase, qpath, gpath, bpath, raw_dim);

    // ---------------- HNSW (cached)
    HNSW g;
    Timer t;
    if (!g.load(cache, ds.xb.data())) {
        g.build(ds.xb.data(), ds.nb, ds.d, M, efC, 42, true);
        g.save(cache);
    }
    printf("[HNSW] ready in %.1fs (build-time graph quality is validated separately)\n", t.sec());
    // NOTE: finalize() materialises the layer-0 CSR and is only needed by the
    // standalone diagnostics (connectivity BFS).  The benchmark path reads the
    // graph through deg()/nbrs(), and building the CSR at 10M nodes is very slow,
    // so it is NOT done here.
    if (cmd == "hnsw") g.finalize();

    if (cmd == "hnsw") {
        // Plaintext HNSW reference curve: Recall@10 (and @20) vs SINGLE-QUERY
        // latency.  Run with --threads 1 to get a true per-query latency instead of
        // an amortised throughput figure, then the network term of a cloud search
        // service can be added on top (1 RTT + result serialisation).
        auto efs_list = split_int(arg(argc, argv, "--efs-list", "50,100,200"));
        const long nq_h = (long)std::min<size_t>((size_t)std::max(1, nq_max), ds.nq);
        std::ofstream hf(out);
        if (hf) {
            hf << "dataset,D,efS,threads,nq,recall@10,recall@20,mean_us_per_query,"
                  "lat_hnsw_compute_ms\n";
        }
        for (int efS : efs_list) {
            long h10 = 0, h20 = 0; Timer tq;
            #pragma omp parallel for schedule(dynamic, 32) reduction(+:h10,h20)
            for (long i = 0; i < nq_h; i++) {
                std::vector<u32> res; g.search(ds.xq.data() + i * ds.d, 20, efS, res);
                const i32* gt = ds.gt.data() + (size_t)i * ds.dgt;
                int a10 = 0, a20 = 0;
                for (int a = 0; a < (int)res.size() && a < 20; a++)
                    for (int j = 0; j < ds.dgt; j++)
                        if ((i32)res[a] == gt[j]) {
                            if (a < 10 && j < 10) a10++;
                            if (j < 20) a20++;
                            break;
                        }
                h10 += a10; h20 += a20;
            }
            const double us = tq.ms() * 1000.0 / nq_h;
            const double r10 = (double)h10 / (10.0 * nq_h);
            const double r20 = (double)h20 / (20.0 * nq_h);
            printf("[HNSW] efSearch=%4d  Recall@10=%.4f  Recall@20=%.4f  (%.1f us/q, %d threads)\n",
                   efS, r10, r20, us, nthreads);
            if (hf)
                hf << (tag.empty() ? "-" : tag) << ',' << ds.d << ',' << efS << ','
                   << nthreads << ',' << nq_h << ',' << fmt(r10, 4) << ',' << fmt(r20, 4)
                   << ',' << fmt(us, 2) << ',' << fmt(us / 1000.0, 4) << "\n";
        }
        if (hf) hf.close();
        printf("[HNSW] written to %s\n", out.c_str());
        return 0;
    }

    // ---------------- E1: completeness / recall-loss attribution diagnostic
    // Where does the recall loss come from?  For the ground-truth top-k of every
    // query we check, in order:
    //   (a) is the vector packed into at least one HGI group?        -> coverage
    //   (b) is one of its groups among the tau routed groups?        -> routing
    //   (c) did it survive the encrypted filtering inside a group?   -> filtering
    //   (d) did it make it into the returned top-k?                  -> ranking
    // Losses are reported as fractions of the total ground-truth items.
    if (cmd == "diag") {
        LinkSim L_loc = LinkSim::local();
        const int k_ev = std::min(k_eval > 0 ? k_eval : 10, ds.dgt);
        std::vector<std::string> dh = {
            "config","m","bps","B","anchor_level","r","tau","nq","gt_k",
            "hgi_groups","hgi_coverage","hgi_replication","avg_groups_per_vector",
            "frac_in_any_group","frac_in_selected","frac_in_cand","frac_in_topk",
            "loss_coverage","loss_routing","loss_filtering","loss_ranking",
            "recall@10"
        };
        std::ofstream df(out);
        for (size_t i = 0; i < dh.size(); i++) df << (i ? "," : "") << dh[i];
        df << "\n"; df.flush();
        auto demit = [&](const std::vector<std::string>& r) {
            for (size_t i = 0; i < r.size(); i++) df << (i ? "," : "") << r[i];
            df << "\n"; df.flush();
        };

        for (int m : m_list) {
            for (int B : bs_list) {
                EHSearch sys;
                sys.ef_route = ef_route; sys.do_crypto = !no_crypto;
                sys.opts = OP;
                sys.build_index(g, ds.xb.data(), ds.nb, B, anchor_level, m, bps_arg, true, hgi_complete);
                if (backend == "ssd") {
                    if (payload_f.empty() || !sys.attach_payload_ssd(payload_f, (u32)ds.d)) return 1;
                } else {
                    sys.build_payload(ds.xb.data(), ds.nb);
                }
                ReverseIndex rev;
                rev.build(sys.hgi, ds.nb);
                double avg_groups = 0;
                for (size_t v = 0; v < ds.nb; v += 997) avg_groups += rev.deg((u32)v);
                avg_groups /= (ds.nb / 997 + 1);
                printf("[diag] reverse index %.0f MB, %.2f groups/vector\n",
                       (double)rev.mem_bytes() / 1048576.0, avg_groups);

                for (int rr : r_list) {
                    for (int tau : tau_list) {
                        sys.r = rr; sys.filter_mode = 0; sys.k = k_max;
                        int nq_use = std::min(nq_max, (int)ds.nq);
                        std::vector<QueryMetrics> Q(nq_use);
                        #pragma omp parallel num_threads(nthreads)
                        {
                            Scratch sc; sc.init(ds.nb, ds.d, sys.m, sys.kper);
                            #pragma omp for schedule(dynamic, 8)
                            for (int i = 0; i < nq_use; i++)
                                sys.query(ds.xq.data() + (size_t)i * ds.d, tau, sc, Q[i], &L_loc, true);
                        }
                        long n_gt = 0, in_any = 0, in_sel = 0, in_cand = 0, in_top = 0;
                        for (int i = 0; i < nq_use; i++) {
                            const i32* gr = ds.gt.data() + (size_t)i * ds.dgt;
                            std::vector<i32> gsel = Q[i].gidx_diag;
                            std::sort(gsel.begin(), gsel.end());
                            std::vector<u32> cnd = Q[i].cand_diag;
                            std::sort(cnd.begin(), cnd.end());
                            for (int j = 0; j < k_ev; j++) {
                                u32 t = (u32)gr[j];
                                n_gt++;
                                size_t dg = rev.deg(t);
                                if (dg > 0) in_any++;
                                bool sel = false;
                                for (size_t z = 0; z < dg; z++)
                                    if (std::binary_search(gsel.begin(), gsel.end(), (i32)rev.groups(t)[z])) { sel = true; break; }
                                if (sel) in_sel++;
                                if (std::binary_search(cnd.begin(), cnd.end(), t)) in_cand++;
                                bool top = false;
                                for (int z = 0; z < k_ev && z < (int)Q[i].result.size(); z++)
                                    if (Q[i].result[z] == t) { top = true; break; }
                                if (top) in_top++;
                            }
                        }
                        const double N = std::max<long>(1, n_gt);
                        double f_any = in_any / N, f_sel = in_sel / N, f_cand = in_cand / N, f_top = in_top / N;
                        demit({
                            "m" + std::to_string(m) + "_B" + std::to_string(B) + "_r" + std::to_string(rr) + "_tau" + std::to_string(tau),
                            std::to_string(m), std::to_string(sys.bps), std::to_string(B),
                            std::to_string(anchor_level), std::to_string(rr), std::to_string(tau),
                            std::to_string(nq_use), std::to_string(k_ev),
                            std::to_string(sys.hgi.anchor.size()), fmt(sys.hgi.coverage, 4), fmt(sys.hgi.replication, 3),
                            fmt(avg_groups, 2),
                            fmt(f_any, 4), fmt(f_sel, 4), fmt(f_cand, 4), fmt(f_top, 4),
                            fmt(1.0 - f_any, 4), fmt(f_any - f_sel, 4), fmt(f_sel - f_cand, 4), fmt(f_cand - f_top, 4),
                            fmt(f_top, 4)
                        });
                        printf("   m=%d B=%d r=%d tau=%2d | cover-loss %.4f  route-loss %.4f  filter-loss %.4f  rank-loss %.4f  => Recall@%d %.4f\n",
                               m, B, rr, tau, 1.0 - f_any, f_any - f_sel, f_sel - f_cand, f_cand - f_top, k_ev, f_top);
                        fflush(stdout);
                    }
                }
            }
        }
        df.close();
        printf("[diag] written to %s\n", out.c_str());
        return 0;
    }

    // ---------------- benchmark grid
    std::vector<std::string> header = {
        "config","d","m","bps","code_bits","B","anchor_level","r","filter","tau","k","nq",
        "recall@1","recall@10","recall@20","recall@100",
        "n_groups","n_cand","req_bytes","resp_bytes","bytes_per_query","bytes_per_query_MB",
        "compute_ms_mean","route_ms","trapgen_ms","server_ms","decrypt_ms","rerank_ms",
        "lat_local_mean_ms","lat_local_p50_ms","lat_local_p95_ms","lat_local_p99_ms",
        "lat_lan_mean_ms","lat_lan_p50_ms","lat_lan_p95_ms","lat_lan_p99_ms",
        "lat_wan_mean_ms","lat_wan_p50_ms","lat_wan_p95_ms","lat_wan_p99_ms",
        "qps_compute_1t","qps_compute_Nt","qps_lan_1client","qps_wan_1client",
        "qps_bw_bound_lan","qps_bw_bound_wan",
        "client_mem_mb","server_mem_mb","server_index_mb","server_payload_mb",
        "client_disk_mb","server_disk_mb",
        "backend","ssd_read_MB_per_query","ssd_pages4k_per_query","ssd_fetch_ms_per_query",
        "opts",
        "proc_read_MB_1t_pass","tag",
        "hgi_groups","hgi_coverage","hgi_replication",
        "hnsw_upper","hnsw_upper_pct","build_index_s","build_mi_s","build_hgi_s"
    };
    std::vector<std::vector<std::string>> rows;
    std::ofstream csvf(out);
    if (csvf) {
        for (size_t i = 0; i < header.size(); i++) csvf << (i ? "," : "") << header[i];
        csvf << "\n"; csvf.flush();
    } else {
        printf("[Warn] cannot open %s for streaming\n", out.c_str());
    }
    auto emit = [&](const std::vector<std::string>& r) {
        rows.push_back(r);
        if (!csvf) return;
        for (size_t i = 0; i < r.size(); i++) csvf << (i ? "," : "") << r[i];
        csvf << "\n"; csvf.flush();
    };

    LinkSim L_lan{"LAN", lan[0], lan[1]};
    LinkSim L_wan{"WAN", wan[0], wan[1]};
    LinkSim L_loc = LinkSim::local();
    printf("[Link] LAN %.2f Gbps / %.1f ms RTT | WAN %.3f Gbps / %.1f ms RTT\n",
           L_lan.bandwidth_gbps, L_lan.rtt_ms, L_wan.bandwidth_gbps, L_wan.rtt_ms);

    for (int m : m_list) {
        for (int B : bs_list) {
            printf("\n================ m=%d B=%d anchor_level=%d ================\n", m, B, anchor_level);
            EHSearch sys;
            sys.ef_route = ef_route;
            sys.do_crypto = !no_crypto;
            sys.opts = OP;
            Timer tb;
            sys.build_index(g, ds.xb.data(), ds.nb, B, anchor_level, m, bps_arg, true, hgi_complete);
            double build_index_s = tb.sec();
            if (backend == "ssd") {
                if (payload_f.empty() || !sys.attach_payload_ssd(payload_f, (u32)ds.d)) {
                    fprintf(stderr, "SSD backend requires a valid --payload-file (got '%s')\n",
                            payload_f.c_str());
                    return 1;
                }
                printf("  [Backend] SSD / out-of-core (mmap payload, globally sorted fetch)\n");
            } else {
                sys.build_payload(ds.xb.data(), ds.nb);
            }

            DiskFootprint fp = measure_disk(sys, write_art, artifact_dir);
            printf("  [Footprint] client %.2f MB | server %.2f MB (index %.2f MB + payload %.2f MB) "
                   "| disk C %.2f MB / S %.2f MB\n",
                   sys.client_mem_mb(), sys.server_mem_mb(),
                   (double)sys.mi.ent.size() * 8 / 1048576.0,
                   (double)sys.srv.payload_bytes() / 1048576.0,
                   fp.client / 1048576.0, fp.server / 1048576.0);

            for (int r : r_list) {
                for (std::string fstr : split(fmodes, '/')) {
                    int mode = (fstr == "pigeonhole") ? 1 : 0;
                    for (int tau : tau_list) {
                        sys.r = r; sys.filter_mode = mode; sys.k = k_max;
                        int nq_use = std::min(nq_max, (int)ds.nq);
                        int nq_s = std::min(nq_1t, nq_use);
                        if (mode == 1 && nq_use > 500) nq_use = 500;   // pigeonhole is a diagnostic mode

                        // ---- pass A: 1 thread (per-query latency breakdown)
                        std::vector<QueryMetrics> A;
                        std::vector<double> a1, a10, a20, a100;
                        double wA = 0;
                        u64 io0 = io_read_bytes();
                        run_pass(sys, ds, tau, nq_s, 1, L_loc, nullptr, k_max, &A, &a1, &a10, &a20, &a100, &wA);
                        double proc_read_MB = (double)(io_read_bytes() - io0) / 1048576.0;
                        // ---- pass B: N threads (throughput)
                        std::vector<QueryMetrics> Bm;
                        std::vector<double> b1, b10, b20, b100;
                        double wB = 0;
                        run_pass(sys, ds, tau, nq_use, nthreads, L_loc, nullptr, k_max,
                                 &Bm, &b1, &b10, &b20, &b100, &wB);

                        auto mean = [](const std::vector<double>& v) {
                            double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); };

                        // latency using pass A compute times + simulated link
                        std::vector<double> lt_loc, lt_lan, lt_wan;
                        double sr = 0, stg = 0, ssv = 0, sdec = 0, sre = 0, scmp = 0, ngr = 0, ncd = 0;
                        u64 req = 0, resp = 0;
                        for (auto& q : A) {
                            lt_loc.push_back(q.compute_ms);
                            lt_lan.push_back(q.compute_ms + L_lan.added_sec(q.req_bytes, q.resp_bytes) * 1e3);
                            lt_wan.push_back(q.compute_ms + L_wan.added_sec(q.req_bytes, q.resp_bytes) * 1e3);
                            sr += q.route_ms; stg += q.trapgen_ms; ssv += q.server_ms;
                            sdec += q.decrypt_ms; sre += q.rerank_ms; scmp += q.compute_ms;
                            ngr += q.n_groups; ncd += q.n_cand;
                            req += q.req_bytes; resp += q.resp_bytes;
                        }
                        double nA = A.empty() ? 1 : (double)A.size();
                        LatStats LL = lat_stats(lt_loc), LLan = lat_stats(lt_lan), LWa = lat_stats(lt_wan);
                        double bpp = (req + resp) / nA;

                        // SSD backend I/O accounting (summed over pass A)
                        u64 io_bytes = 0, io_pages = 0; double io_ms = 0;
                        for (auto& q : A) { io_bytes += q.io.bytes_fetched; io_pages += q.io.pages_4k;
                                            io_ms += q.io.fetch_ms; }

                        double qps_c1 = (wA > 0) ? nq_s / wA : 0;
                        double qps_cN = (wB > 0) ? nq_use / wB : 0;
                        double qps_lan1 = LLan.mean > 0 ? 1000.0 / LLan.mean : 0;
                        double qps_wan1 = LWa.mean > 0 ? 1000.0 / LWa.mean : 0;
                        double qps_bw_lan = bpp > 0 ? (L_lan.bandwidth_gbps * 1e9 / 8.0) / bpp : 0;
                        double qps_bw_wan = bpp > 0 ? (L_wan.bandwidth_gbps * 1e9 / 8.0) / bpp : 0;

                        double r1 = mean(b1), r10 = mean(b10), r20 = mean(b20), r100 = mean(b100);
                        // quick sanity line
                        printf("   r=%d %-10s tau=%2d | Rec@20=%.3f @10=%.3f | cand=%6.0f | "
                               "loc %.2f | LAN %.2f (p95 %.2f) | WAN %.2f (p95 %.2f) ms | "
                               "bw %.0f KB | QPSc1 %6.0f QPSn %7.0f\n",
                               r, fstr.c_str(), tau, r20, r10, ncd / nA, LL.mean,
                               LLan.mean, LLan.p95, LWa.mean, LWa.p95, bpp / 1024.0, qps_c1, qps_cN);
                        fflush(stdout);

                        emit({
                            "m" + std::to_string(m) + "_B" + std::to_string(B) + "_a" + std::to_string(anchor_level)
                                + "_r" + std::to_string(r) + "_" + fstr,
                            std::to_string(ds.d),
                            std::to_string(m), std::to_string(sys.bps),
                            std::to_string(std::min(ds.d, m * sys.bps)),
                            std::to_string(B),
                            std::to_string(anchor_level), std::to_string(r), fstr,
                            std::to_string(tau), std::to_string(k_max), std::to_string(nq_use),
                            fmt(r1,4), fmt(r10,4), fmt(r20,4), fmt(r100,4),
                            fmt(ngr / nA, 1), fmt(ncd / nA, 1),
                            std::to_string(req / (u64)nA), std::to_string(resp / (u64)nA),
                            fmt(bpp, 1), fmt(bpp / 1048576.0, 4),
                            fmt(scmp / nA, 4), fmt(sr / nA, 4), fmt(stg / nA, 4), fmt(ssv / nA, 4),
                            fmt(sdec / nA, 4), fmt(sre / nA, 4),
                            fmt(LL.mean,4), fmt(LL.p50,4), fmt(LL.p95,4), fmt(LL.p99,4),
                            fmt(LLan.mean,4), fmt(LLan.p50,4), fmt(LLan.p95,4), fmt(LLan.p99,4),
                            fmt(LWa.mean,4), fmt(LWa.p50,4), fmt(LWa.p95,4), fmt(LWa.p99,4),
                            fmt(qps_c1,1), fmt(qps_cN,1), fmt(qps_lan1,1), fmt(qps_wan1,1),
                            fmt(qps_bw_lan,1), fmt(qps_bw_wan,1),
                            fmt(sys.client_mem_mb(),2), fmt(sys.server_mem_mb(),2),
                            fmt((double)sys.mi.ent.size()*8/1048576.0,2),
                            fmt((double)sys.srv.payload_bytes()/1048576.0,2),
                            fmt(fp.client/1048576.0,2), fmt(fp.server/1048576.0,2),
                            sys.srv.ssd_mode() ? "ssd" : "mem",
                            fmt(io_bytes / nA / 1048576.0, 4),
                            fmt(io_pages / nA, 1),
                            fmt(io_ms / nA, 4),
                            optstr.empty() ? "none" : plus_join(optstr),
                            fmt((double)proc_read_MB, 2),
                            tag.empty() ? "-" : tag,
                            std::to_string(sys.hgi.anchor.size()), fmt(sys.hgi.coverage,4),
                            fmt(sys.hgi.replication,3),
                            std::to_string(g.count_upper(0)),
                            fmt(100.0*g.count_upper(0)/g.n,3),
                            fmt(build_index_s,2), fmt(sys.mi.build_sec,2), fmt(sys.hgi.build_sec,2)
                        });
                        std::vector<QueryMetrics>().swap(Bm);
                        if (mode == 1) break;   // one diagnostic point is enough
                    }
                }
            }
        }
    }
    if (csvf) csvf.close();
    printf("\n[Result] %zu configurations written to %s\n", rows.size(), out.c_str());
    printf("[Mem] process peak RSS = %.0f MB\n", peak_rss_mb());
    return 0;
}
