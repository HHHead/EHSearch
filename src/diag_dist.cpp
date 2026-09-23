// diag_dist.cpp - validate l2_sqr against a scalar reference
#include "common.h"
#include "simd.h"
using namespace eh;
int main(int argc, char** argv) {
    std::string dir = "./data/sift";
    size_t nb, nq; int d;
    auto xb = read_fvecs(dir + "/sift_base.fvecs", &nb, &d);
    auto xq = read_fvecs(dir + "/sift_query.fvecs", &nq, &d);
    size_t ng; int dgt;
    auto gt = read_ivecs(dir + "/sift_groundtruth.ivecs", &ng, &dgt);
    print_cpu_caps();
    printf("d=%d nb=%zu\n", d, nb);
    auto ref = [&](const float* a, const float* b) {
        double s = 0; for (int i = 0; i < d; i++) { double t = (double)a[i]-b[i]; s += t*t; } return s;
    };
    // a few pairs
    for (int t = 0; t < 3; t++) {
        int qi = t, bi = (int)gt[(size_t)t*dgt + 0];
        float a = l2_sqr(xq.data()+(size_t)qi*d, xb.data()+(size_t)bi*d, d);
        double r = ref(xq.data()+(size_t)qi*d, xb.data()+(size_t)bi*d);
        printf("pair(q%d, gt NN %d): simd=%.4f ref=%.4f  diff=%.4f\n", qi, bi, a, r, a-r);
    }
    // full correlation on a sample: top-20 by simd vs ref for 3 queries
    for (int qi = 0; qi < 3; qi++) {
        const float* q = xq.data()+(size_t)qi*d;
        std::vector<std::pair<float,int>> vs(nb), vr(nb);
        for (size_t j = 0; j < nb; j++) {
            vs[j] = {l2_sqr(q, xb.data()+j*(size_t)d, d), (int)j};
            vr[j] = {(float)ref(q, xb.data()+j*(size_t)d), (int)j};
        }
        std::partial_sort(vs.begin(), vs.begin()+20, vs.end());
        std::partial_sort(vr.begin(), vr.begin()+20, vr.end());
        int ov = 0; for (int a = 0; a < 20; a++) for (int b = 0; b < 20; b++)
            if (vs[a].second == vr[b].second) { ov++; break; }
        printf("query %d: simd top1=%d(%.2f) ref top1=%d(%.2f) ; overlap@20 = %d/20 ; "
               "official gt[0]=%d matches ref_top1=%d matches simd_top1=%d\n",
               qi, vs[0].second, vs[0].first, vr[0].second, vr[0].first, ov,
               gt[(size_t)qi*dgt], vr[0].second==gt[(size_t)qi*dgt], vs[0].second==gt[(size_t)qi*dgt]);
        fflush(stdout);
    }
    return 0;
}
