# EHSearch -- Reference Implementation

**Anonymous artifact for double-blind review.**
This repository is released only to support the evaluation of the submitted paper.
It deliberately contains no author names, affiliations, institutional e-mail addresses,
or machine-specific paths. Please do not attempt to identify the authors.

---

## 1. What this is

EHSearch is a **single-round private proximity search** system. A client holds a query
vector; a server holds an encrypted copy of the database plus a graph-distilled,
group-scoped inverted index. One request/response round is enough to answer a query:
the client sends a trapdoor, the server replies with the matched ciphertexts, and the
client decrypts and re-ranks them exactly.

This repository contains the **C++ reference implementation** of the core system:

* the offline **index builder** (graph topology, binary quantization, group-scoped
  multi-index, encrypted payload file), and
* the online **query pipeline** used to produce every number in the paper.

Auxiliary diagnostic, self-test and tuning tools are included so that the structural
claims in the paper can be re-checked locally.

The system is **not** a general-purpose search library. It is a research prototype
intended to make the paper's results reproducible.

---

## 2. Repository layout

```
.
|-- Makefile                     build rules (main driver + tools)
|-- README.md                    this file
|-- LICENSE
|-- src/
|   |-- common.h                 types, fvecs/ivecs IO, timers, memory accounting,
|   |                            link simulation (LAN/WAN models)
|   |-- simd.h                   SIMD kernels: L2 / inner product / popcount /
|   |                            batched rotation / top-k selection, CPU capability
|   |                            detection and runtime dispatch
|   |-- kernels.cpp              out-of-line instantiations of the SIMD kernels
|   |-- crypto.h                 AES-128 with AES-NI / VAES, and the AES-based PRF
|   |                            used for table keys, object ids and trapdoors
|   |-- hnsw.h                   self-contained HNSW (build, plaintext search,
|   |                            layer-0 CSR export for diagnostics)
|   |-- rabitq.h                 randomized binary quantization: orthogonal
|   |                            rotation, sign binarization, segment packing
|   |-- hgi.h                    Topology-Distilled Index: group construction,
|   |                            group-scoped multi-index, the two filter routines
|   |-- ssd.h                    out-of-core payload backend: zero-copy mmap and
|   |                            globally sorted fetch
|   |-- ehsearch.h               client/server pipeline for both backends,
|   |                            per-query timing and traffic accounting
|   |-- main.cpp                 benchmark driver: `hnsw`, `prepare`, `bench`
|   |-- diag_graph.cpp           connectivity / coverage diagnostics on the graph
|   |-- diag_search.cpp          brute-force sanity check against ground truth
|   |-- diag_dist.cpp            SIMD kernels vs. scalar reference
|   |-- diag_dump.cpp            raw value dump for cross-implementation checks
|   |-- test_hnsw.cpp            HNSW build/search self-test
|   |-- test_plain.cpp           plaintext end-to-end self-test
|   |-- tune_hnsw.cpp            efConstruction / M sweep helper
|   `-- tune_scale.cpp           out-of-core scale sweep helper
`-- scripts/
    `-- example_run.sh           minimal end-to-end walkthrough
```

---

## 3. Requirements

| Item | Requirement |
|---|---|
| Compiler | A C++17 compiler. Tested with GCC; MinGW-w64 and Clang are expected to work |
| CPU | x86-64. AES-NI is **required** for the crypto path (`-maes`); VAES, AVX2 and AVX-512 are used when present, detected at run time |
| Runtime | OpenMP (`-fopenmp`) |
| OS | Linux and Windows (MinGW). The OS-specific parts (process memory counters, `mmap`) are behind `#ifdef _WIN32` |
| Third-party libraries | **None.** Only the C++ standard library, OpenMP, compiler intrinsics and OS primitives |

> The code is compiled with `-maes`, so a CPU without AES-NI will fault at run time.
> For a crypto-free measurement of the search path, pass `--no-crypto` (see below).

---

## 4. Build

```bash
make            # -> build/ehsearch plus the auxiliary tools
make main       # -> build/ehsearch only
make -j4        # parallel build
make clean
```

Override any part of the toolchain, for example:

```bash
make CXX=clang++ OPT="-O3 -march=znver3 -DNDEBUG"
```

---

## 5. Data format

The driver reads the standard **fvecs / ivecs** format used by public ANN benchmark
suites, so no conversion step is needed.

| File | Meaning |
|---|---|
| `*_base.fvecs` | database vectors, `int32 d` followed by `d` floats per row |
| `*_query.fvecs` | query vectors, same layout |
| `*_groundtruth.ivecs` | exact top-*k* neighbour ids, `int32 k` followed by `k` int32 per row |

Any dimension is supported: the dimension is taken from the file header. A reduced
prefix of the database can be requested with `--nbase N`, which is how the scale
sweeps are produced.

---

## 6. Quick start

```bash
# 0) place data, e.g.
#    data/sift/sift_base.fvecs  data/sift/sift_query.fvecs
#    data/sift/sift_groundtruth.ivecs

# 1) build the graph topology once and cache it
./build/ehsearch hnsw \
    --base  data/sift/sift_base.fvecs \
    --cache build/hnsw_sift.bin \
    --M 32 --efC 64

# 2) (out-of-core only) materialise the encrypted payload file
./build/ehsearch prepare \
    --base-raw data/sift/sift_base.fvecs --nbase 1000000 \
    --payload-file build/payload.bin

# 3) run the benchmark
./build/ehsearch bench \
    --base  data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --gt    data/sift/sift_groundtruth.ivecs \
    --cache build/hnsw_sift.bin \
    --m-list 16 --bps 8 --bs-list 3000 --r-list 1 \
    --tau-list 1,5,10,20,30,40,50 \
    --anchor-level 1 --k 100 --k-eval 10 \
    --tag sift --out results/sift.csv
```

Grid arguments accept either a single value or a comma-separated list, so one
invocation sweeps a whole parameter family and writes one CSV row per configuration.
`scripts/example_run.sh` wraps exactly the three steps above.

For the out-of-core backend, add:

```bash
    --backend ssd --payload-file build/payload.bin
```

---

## 7. Command reference

### `hnsw`
Builds the graph topology once and caches it to `--cache`. Re-running loads the
cache instead of rebuilding, and the graph parameters are read back from the cache
header, so `--M` / `--efC` only take effect on the first build.

| Option | Default | Meaning |
|---|---|---|
| `--base` | `./data/sift` | database `.fvecs` file (or a directory) |
| `--cache` | `./build/hnsw_M32_efC64.bin` | graph cache path |
| `--M` | 32 | out-degree per layer (layer 0 uses `2M`) |
| `--efC` | 64 | `efConstruction` |
| `--threads` | 32 | worker threads |

### `prepare`
Writes the encrypted payload file used by the SSD backend, so that the online path
never touches plaintext vectors.

| Option | Meaning |
|---|---|
| `--base-raw` | database `.fvecs` file |
| `--nbase` | number of vectors to include |
| `--dim` | raw dimension (when the header is not authoritative) |
| `--payload-file` | output ciphertext file |

### `bench`
The measurement driver. Emits one CSV row per configuration.

| Option | Default | Meaning |
|---|---|---|
| `--base` / `--query` / `--gt` | -- | database, queries, ground truth |
| `--cache` | `./build/hnsw_M32_efC64.bin` | graph cache from `hnsw` |
| `--m-list` | 8 | sub-segment counts to sweep |
| `--bps` | 0 | bits per sub-segment; `0` derives `d/m` |
| `--bs-list` | 1500 | group capacities to sweep |
| `--r-list` | 0 | Hamming radii to sweep |
| `--tau-list` | `1,5,10,20,30,40,50` | number of routed groups to probe |
| `--anchor-level` | 1 | minimum HNSW level for a group anchor |
| `--ef-route` | 32 | `efSearch` for the routing step |
| `--filter` | union | `union` or `pigeonhole` |
| `--k` | 100 | result list length |
| `--k-eval` | 10 | `k` used for the reported recall (`Recall@k_eval`) |
| `--nq` / `--nq-1t` | 10000 | query counts for the parallel / single-thread passes |
| `--threads` | 32 | worker threads |
| `--lan` / `--wan` | `3.0,1.0` / `0.4,80.0` | link model: `Gbps,RTT_ms` |
| `--backend` | mem | `mem` or `ssd` |
| `--payload-file` | -- | ciphertext file for the SSD backend |
| `--opts` | -- | comma list of `zerocopy`, `omp_groups`, `prefetch`, `inner`, `all` |
| `--out` | `./results/bench.csv` | output CSV |
| `--tag` | -- | free-form label carried into the CSV |

Boolean switches: `--hgi-only` (build the index and print structural statistics, then
exit), `--hgi-complete` (run the coverage-completion pass described below),
`--no-crypto` (skip encryption/decryption to isolate the search cost),
`--write-artifacts` (also write per-configuration footprint artefacts).

### Auxiliary tools
`diag_graph`, `diag_search`, `diag_dist`, `diag_dump`, `test_hnsw`, `test_plain`,
`tune_hnsw`, `tune_scale`. Each is a standalone binary; run it without arguments to
see its usage. They are provided for reproducing the structural checks in the paper
(graph connectivity, index coverage, kernel correctness against a scalar reference)
and are not needed for the main measurements.

---

## 8. Parameters

| Symbol | Flag | Role |
|---|---|---|
| `m` | `--m-list` | number of sub-segments a binary code is split into |
| `bps` | `--bps` | width of one sub-segment in bits; the number of bits actually used is `min(d, m.bps)` |
| `B` | `--bs-list` | group capacity: how many database points each group stores |
| `r` | `--r-list` | Hamming radius accepted per sub-segment (`0` exact, `1` one bit flip, ...) |
| `tau` | `--tau-list` | how many routed groups are probed per query; the main recall/latency knob |
| `anchor_level` | `--anchor-level` | minimum graph level a group anchor must have |

`(m, bps)` controls the matching granularity of the filter, `B` trades index size
against how concentrated candidates are, and `tau` moves along the recall/latency
curve. `--hgi-only` prints the group coverage and replication factor without running
any query, which is the cheapest way to see how `B` affects index structure.

**Coverage.** Group construction walks layer-0 edges from every anchor; groups that
cannot be filled from their connected component are padded. Increasing `B` alone does
**not** guarantee full coverage, because once `B` exceeds the reach of the walk the
groups fill straight from the largest connected component and the padding stops
firing, so points outside that component are never grouped. `--hgi-complete` adds a
completion pass that rewrites duplicated padding slots with the still-uncovered ids,
which makes coverage exactly `1.0` at no extra storage cost.

---

## 9. The two server backends

**In-memory (`--backend mem`).** The server keeps the encrypted payloads and the
multi-index resident. The `tau` routed groups are filtered by one OpenMP thread each
into private buckets, which are merged afterwards through an epoch-stamped visited
array, so the intra-query critical path contains no lock, atomic or barrier.

**Out-of-core (`--backend ssd`).** The prepared ciphertext file is mapped read-only
into virtual memory (`mmap` on POSIX, `MapViewOfFile` on Windows); the server issues
no explicit `read`. Because matched ids arrive in table order, they are globally
sorted into ascending id order before the payload file is touched. Both backends store
records in ascending id order, so the access pattern becomes a monotone sweep over the
file; a software prefetch six records ahead hides the residual latency. The volume
actually pulled from the device is reported from process I/O counters rather than
modelled.

---

## 10. What the CSV contains

Every `bench` row reports, for one configuration:

* **quality** -- `recall@1`, `recall@10`, `recall@20`, `recall@100`, candidate-set size
  `n_cand`, and the group-coverage / replication statistics;
* **cost** -- a latency breakdown in milliseconds
  (`route_ms`, `trapgen_ms`, `server_ms`, `decrypt_ms`, `rerank_ms`,
  `compute_ms_mean`) plus `lat_local_*`, `lat_lan_*`, `lat_wan_*` percentiles, and
  throughput (`qps_compute_1t`, `qps_compute_Nt`, single-client LAN/WAN QPS,
  bandwidth-bound QPS);
* **traffic** -- `req_bytes`, `resp_bytes`, `bytes_per_query`;
* **footprint** -- `server_index_mb`, `server_payload_mb`, `server_mem_mb`,
  `client_mem_mb`.

Recall is always computed against the official ground truth with the first
`--k-eval` entries of the returned list, so that a long result list cannot inflate the
reported value.

---

## 11. Licence

See `LICENSE`. The artifact is released for academic review purposes.
