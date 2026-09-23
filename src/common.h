// common.h - basic types, IO, timing, system metrics, network link simulation
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace eh {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr double NS_PER_S = 1e9;

// ------------------------------------------------------------------ timer
struct Timer {
    std::chrono::steady_clock::time_point t0;
    void reset() { t0 = std::chrono::steady_clock::now(); }
    Timer() { reset(); }
    double sec() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    double ms() const { return sec() * 1e3; }
};

// ------------------------------------------------------------------ IO
// .fvecs / .bvecs : [int32 dim][data...] per record
// .ivecs         : [int32 dim][int32...] per record
// NOTE: after sniffing the dimension from the first 4 bytes we MUST rewind to
// byte 0, because each record is (dim+1) int32 wide *including* its own header.
// Forgetting the rewind shifts the whole array by 4 bytes (one float) and
// silently corrupts every downstream distance computation.
inline std::vector<float> read_fvecs(const std::string& path, size_t* n_out = nullptr, int* d_out = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[IO] cannot open %s\n", path.c_str()); exit(1); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    int32_t d = 0;
    f.read((char*)&d, 4);
    f.seekg(0);                                  // <-- rewind (see note above)
    size_t rec = (size_t)(d + 1) * 4;
    size_t n = bytes / rec;
    std::vector<float> out((size_t)n * d);
    std::vector<char> buf(rec);
    for (size_t i = 0; i < n; i++) {
        f.read(buf.data(), (std::streamsize)rec);
        memcpy(out.data() + i * d, buf.data() + 4, (size_t)d * 4);
    }
    if (n_out) *n_out = n;
    if (d_out) *d_out = d;
    return out;
}

// SIFT-1B style .bvecs written as uint8 payload with int32 dim header
inline std::vector<float> read_bvecs_as_float(const std::string& path, size_t* n_out = nullptr, int* d_out = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[IO] cannot open %s\n", path.c_str()); exit(1); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    int32_t d = 0;
    f.read((char*)&d, 4);
    f.seekg(0);                                  // <-- rewind
    size_t rec = (size_t)d + 4;
    size_t n = bytes / rec;
    std::vector<float> out((size_t)n * d);
    std::vector<u8> buf(rec);
    for (size_t i = 0; i < n; i++) {
        f.read((char*)buf.data(), (std::streamsize)rec);
        for (int j = 0; j < d; j++) out[i * d + j] = (float)buf[4 + j];
    }
    if (n_out) *n_out = n;
    if (d_out) *d_out = d;
    return out;
}

inline std::vector<i32> read_ivecs(const std::string& path, size_t* n_out = nullptr, int* d_out = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[IO] cannot open %s\n", path.c_str()); exit(1); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    int32_t d = 0;
    f.read((char*)&d, 4);
    f.seekg(0);                                  // <-- rewind
    size_t rec = (size_t)(d + 1) * 4;
    size_t n = bytes / rec;
    std::vector<i32> out((size_t)n * d);
    std::vector<char> buf(rec);
    for (size_t i = 0; i < n; i++) {
        f.read(buf.data(), (std::streamsize)rec);
        memcpy(out.data() + i * d, buf.data() + 4, (size_t)d * 4);
    }
    if (n_out) *n_out = n;
    if (d_out) *d_out = d;
    return out;
}

inline u64 file_size(const std::string& path) {
    std::error_code ec;
    auto s = std::filesystem::file_size(path, ec);
    return ec ? 0 : (u64)s;
}
inline u64 file_size_or_dir(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        u64 tot = 0;
        for (auto& e : std::filesystem::recursive_directory_iterator(path, ec))
            if (e.is_regular_file()) tot += e.file_size();
        return tot;
    }
    return file_size(path);
}

// ------------------------------------------------------------------ system metrics
#ifdef _WIN32
inline double rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / 1048576.0;
    return -1;
}
inline double peak_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.PeakWorkingSetSize / 1048576.0;
    return -1;
}
#else
inline double rss_mb() {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long pages = 0, res = 0;
    if (fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0;
    fclose(f);
    return (double)res * sysconf(_SC_PAGESIZE) / 1048576.0;
}
inline double peak_rss_mb() { return rss_mb(); }
#endif

// ------------------------------------------------------------------ latency stats
struct LatStats {
    double mean = 0, p50 = 0, p90 = 0, p95 = 0, p99 = 0, mn = 0, mx = 0;
};
inline LatStats lat_stats(std::vector<double> v) {
    LatStats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / v.size();
    auto q = [&](double p) {
        size_t i = (size_t)std::min<double>(v.size() - 1, std::max<double>(0, p * (v.size() - 1)));
        return v[i];
    };
    s.p50 = q(0.50); s.p90 = q(0.90); s.p95 = q(0.95); s.p99 = q(0.99);
    s.mn = v.front(); s.mx = v.back();
    return s;
}

// ------------------------------------------------------------------ network link simulation
// Models a single client<->server link.
//   latency = RTT + serialization(req) + serialization(resp) + compute
//   bandwidth(bps) and rtt(s) taken from Compass' setting as requested:
//     LAN : 3 Gbps  bandwidth, 1 ms round-trip latency
//     WAN : 400 Mbps bandwidth, 80 ms round-trip latency
struct LinkSim {
    std::string name = "local";
    double bandwidth_gbps = 0.0;   // 0 => no network cost at all
    double rtt_ms = 0.0;

    double serial_sec(u64 bytes) const {
        if (bandwidth_gbps <= 0) return 0.0;
        return (double)bytes * 8.0 / (bandwidth_gbps * 1e9);
    }
    // total added seconds for a 1-RTT protocol exchanging the given byte counts
    double added_sec(u64 req_bytes, u64 resp_bytes) const {
        if (bandwidth_gbps <= 0) return 0.0;
        return rtt_ms * 1e-3 + serial_sec(req_bytes) + serial_sec(resp_bytes);
    }

    static LinkSim local() { return LinkSim{"local", 0.0, 0.0}; }
    static LinkSim lan()   { return LinkSim{"LAN",   3.0, 1.0}; }   // 3 Gbps, 1 ms
    static LinkSim wan()   { return LinkSim{"WAN",   0.4, 80.0}; }  // 400 Mbps, 80 ms
};

// ------------------------------------------------------------------ csv
inline void write_csv(const std::string& path, const std::vector<std::string>& header,
                      const std::vector<std::vector<std::string>>& rows) {
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "[CSV] cannot write %s\n", path.c_str()); return; }
    for (size_t i = 0; i < header.size(); i++) f << (i ? "," : "") << header[i];
    f << "\n";
    for (auto& r : rows) {
        for (size_t i = 0; i < r.size(); i++) f << (i ? "," : "") << r[i];
        f << "\n";
    }
}
inline std::string fmt(double x, int prec = 4) {
    char b[64]; snprintf(b, sizeof(b), "%.*f", prec, x); return b;
}

// ------------------------------------------------------------------ real device I/O counters
// Bytes actually pulled from the storage device (page faults included), used to
// cross-check the modelled "payload volume" of the SSD backend.
inline u64 io_read_bytes() {
#ifdef _WIN32
    IO_COUNTERS c{};
    if (GetProcessIoCounters(GetCurrentProcess(), &c)) return (u64)c.ReadTransferCount;
    return 0;
#else
    FILE* f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256]; u64 rd = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "read_bytes: %llu", (unsigned long long*)&rd) == 1) break;
    fclose(f);
    return rd;
#endif
}

// ------------------------------------------------------------------ raw vector readers
// bigann-style raw uint8 blob without per-record headers (SIFT-1B / sift1b_raw_uint8.bin)
inline std::vector<float> read_raw_u8_as_float(const std::string& path, size_t n, int d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[IO] cannot open %s\n", path.c_str()); exit(1); }
    std::vector<u8> buf((size_t)n * d);
    f.read((char*)buf.data(), (std::streamsize)buf.size());
    if ((size_t)f.gcount() != buf.size()) {
        fprintf(stderr, "[IO] %s too short for %zu x %d\n", path.c_str(), n, d);
        exit(1);
    }
    std::vector<float> out(buf.size());
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)buf.size(); i++) out[i] = (float)buf[i];
    return out;
}

// raw little-endian float32 blob without headers (e.g. sift10m_float32.bin)
inline std::vector<float> read_raw_f32(const std::string& path, size_t n, int d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[IO] cannot open %s\n", path.c_str()); exit(1); }
    std::vector<float> out((size_t)n * d);
    f.read((char*)out.data(), (std::streamsize)out.size() * 4);
    if ((size_t)f.gcount() != out.size() * 4) {
        fprintf(stderr, "[IO] %s too short for %zu x %d\n", path.c_str(), n, d);
        exit(1);
    }
    return out;
}

} // namespace eh
