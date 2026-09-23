// diag_dump.cpp - dump raw values so they can be compared against numpy
#include "common.h"
using namespace eh;
int main() {
    std::string dir = "./data/sift";
    size_t nb, nq; int d, dq;
    auto xb = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &dq);
    size_t ng; int dgt;
    auto gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &ng, &dgt);
    printf("C++ nb=%zu d=%d nq=%zu dq=%d ng=%zu dgt=%d\n", nb, d, nq, dq, ng, dgt);
    auto dump = [&](const char* tag, const float* p) {
        printf("%s:", tag);
        for (int i = 0; i < 8; i++) printf(" %g", p[i]);
        printf("\n");
    };
    dump("base[0]      ", xb.data());
    dump("base[1]      ", xb.data() + d);
    dump("base[932085] ", xb.data() + (size_t)932085 * d);
    dump("base[934876] ", xb.data() + (size_t)934876 * d);
    dump("query[0]     ", xq.data());
    printf("gt[0][0..4]: %d %d %d %d %d\n", gt[0], gt[1], gt[2], gt[3], gt[4]);
    printf("file bytes base=%llu query=%llu gt=%llu\n",
        (unsigned long long)file_size(dir + "/sift_base.fvecs"),
        (unsigned long long)file_size(dir + "/sift_query.fvecs"),
        (unsigned long long)file_size(dir + "/sift_groundtruth.ivecs"));
    return 0;
}
