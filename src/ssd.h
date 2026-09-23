// ssd.h - out-of-core payload backend: zero-copy mmap + globally sorted fetch
//
// Mirrors the paper's SSD variant (Section 8 "2) SSD" / Section 9.5) and the
// reference prototype `ssd/EHSearch-ssd.py`:
//   * the encrypted vector payload lives in one file on disk and is mapped into
//     virtual memory (no explicit read() calls, the OS handles page faults),
//   * matched candidate ids are globally sorted before the fetch so that the
//     fragmented random reads become a monotone sweep over the file,
//   * the fetched records are permuted back into the original candidate order so
//     the client does not need to receive a separate id list (the network payload
//     stays identical to the in-memory backend).
#pragma once
#include "common.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace eh {

// ------------------------------------------------------------------ mmap
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile() { close(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    bool open(const std::string& path, bool random_access = true) {
        close();
#ifdef _WIN32
        DWORD flags = FILE_ATTRIBUTE_NORMAL | (random_access ? FILE_FLAG_RANDOM_ACCESS : 0);
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, flags, nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE) { hFile_ = INVALID_HANDLE_VALUE; return false; }
        LARGE_INTEGER li{};
        if (!GetFileSizeEx(hFile_, &li)) return false;
        size_ = (size_t)li.QuadPart;
        hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap_) return false;
        base_ = MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0);
        if (!base_) return false;
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;
        struct stat st{};
        if (fstat(fd_, &st) != 0) return false;
        size_ = (size_t)st.st_size;
        base_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) { base_ = nullptr; return false; }
        ::madvise(base_, size_, random_access ? MADV_RANDOM : MADV_SEQUENTIAL);
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (base_) { UnmapViewOfFile(base_); base_ = nullptr; }
        if (hMap_) { CloseHandle(hMap_); hMap_ = nullptr; }
        if (hFile_ != INVALID_HANDLE_VALUE) { CloseHandle(hFile_); hFile_ = INVALID_HANDLE_VALUE; }
#else
        if (base_) { ::munmap(base_, size_); base_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        size_ = 0;
    }

    const u8* base() const { return (const u8*)base_; }
    size_t size() const { return size_; }
    bool ok() const { return base_ != nullptr; }

    inline const u8* record(size_t idx, size_t stride) const { return base() + idx * stride; }

private:
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMap_ = nullptr;
#else
    int fd_ = -1;
#endif
    void* base_ = nullptr;
    size_t size_ = 0;
};

// ------------------------------------------------------------------ I/O stats
struct IoStats {
    u64 bytes_fetched = 0;    // modelled payload volume (|C| * stride)
    u64 pages_4k = 0;         // distinct 4 KiB pages touched by the fetch
    double fetch_ms = 0.0;    // wall time of the fetch (includes page faults)
    u64 process_read_bytes = 0;  // real bytes pulled from the device (process I/O counters)
};

// ------------------------------------------------------------------ payload store
// Two backends with identical query-side semantics:
//   MEM : std::vector<u8> resident in RAM
//   SSD : read-only mmap over a file, fetched in sorted-id order
struct PayloadStore {
    bool ssd = false;
    u32 stride = 0, d = 0;
    std::vector<u8> mem;
    MappedFile map;

    bool open_ssd(const std::string& path, u32 stride_, u32 d_) {
        ssd = true; stride = stride_; d = d_;
        if (!map.open(path, /*random_access=*/true)) return false;
        printf("[SSD] mapped %s : %.2f GB, stride %u B/vector\n",
               path.c_str(), (double)map.size() / 1e9, stride);
        return true;
    }

    size_t bytes() const { return ssd ? map.size() : mem.size(); }
    double mem_mb() const { return ssd ? 0.0 : (double)mem.size() / 1048576.0; }

    // Fetch candidate records.  `buf` receives the records in the SAME order as
    // `cand`; internally the reads are issued in ascending id order.
    void fetch(const std::vector<u32>& cand, std::vector<u8>& buf, IoStats* st = nullptr) const {
        const size_t n = cand.size();
        buf.resize(n * (size_t)stride);
        if (n == 0) return;
        const u8* src = ssd ? map.base() : mem.data();
        Timer t;
        std::vector<u32> ord(n);
        for (u32 i = 0; i < n; i++) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](u32 a, u32 b) { return cand[a] < cand[b]; });

        u64 pages = 0;
        u32 last_page = 0xFFFFFFFFu;
        for (u32 k = 0; k < n; k++) {
            const u32 i = ord[k];
            const u64 off = (u64)cand[i] * stride;
            memcpy(buf.data() + (size_t)i * stride, src + off, stride);
            const u32 pg = (u32)(off >> 12);
            if (pg != last_page) { pages++; last_page = pg; }
        }
        if (st) {
            st->bytes_fetched += (u64)n * stride;
            st->pages_4k += pages;
            st->fetch_ms += t.ms();
        }
    }
};

} // namespace eh
