#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <execution>

#include <omp.h>

#include "include/SampleMerge.hpp"

static SMTimeMetrics run_sample_merge_dispatch(int*& start, int*& end, int threads) {
    switch (threads) {
        case 1: return sampleMerge<1>(start, end);
        case 2: return sampleMerge<2>(start, end);
        case 3: return sampleMerge<3>(start, end);
        case 4: return sampleMerge<4>(start, end);
        case 5: return sampleMerge<5>(start, end);
        case 6: return sampleMerge<6>(start, end);
        case 7: return sampleMerge<7>(start, end);
        case 8: return sampleMerge<8>(start, end);
        case 9: return sampleMerge<9>(start, end);
        case 10: return sampleMerge<10>(start, end);
        case 11: return sampleMerge<11>(start, end);
        case 12: return sampleMerge<12>(start, end);
        case 16: return sampleMerge<16>(start, end);
        case 32: return sampleMerge<32>(start, end);
        case 36: return sampleMerge<32>(start, end);
        case 48: return sampleMerge<48>(start, end);
        case 64: return sampleMerge<64>(start, end);
        // case 128: return sampleMerge<128>(start, end);
        // case 256: return sampleMerge<256>(start, end);
        default: return sampleMerge<8>(start, end);
    }
}

static std::vector<int> make_uniform_data(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    std::vector<int> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static void require_sorted_equal(const std::vector<int>& got, const std::vector<int>& expected, const char* label) {
    if (got.size() != expected.size() || got != expected) {
        std::cerr << "FAIL: " << label << " (mismatch vs std::sort)\n";
        std::exit(1);
    }
}

int main() {
    omp_set_dynamic(0);
    std::cin.tie(nullptr);
    std::ios::sync_with_stdio(false);

    // Prueba 1: Correctitud (casos chicos + aleatorio)
    for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(128), size_t(10'000), size_t(1'000'000)}) {
        auto input = make_uniform_data(n, 42u + static_cast<uint32_t>(n));
        auto ref = input;
        std::sort(ref.begin(), ref.end());

        auto work = input;
        int* start = work.data();
        int* end = work.data() + work.size();
        
        if (work.size() >= 1024) {
            omp_set_num_threads(4);
            (void)run_sample_merge_dispatch(start, end, 4);
        } else {
            std::sort(work.begin(), work.end());
        }

        require_sorted_equal(work, ref, "sampleMerge correctitud");
    }
    std::cout << "OK: correctitud\n";

    // Prueba 2: Smoke timing (un tamaño, pocos hilos)
    const size_t n = 200'000'013u; // 1u << 20; // ~1M
    auto base = make_uniform_data(n, 123);
    std::cout << "\nTiming N=" << n << "\n";

    {
        auto v = base;
        auto t0 = std::chrono::high_resolution_clock::now();
        std::sort(std::execution::par_unseq, v.begin(), v.end());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "std::sort (par_unseq):   " << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    for (int threads : {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 16, 32, 36, 48, 64}) {
        auto v = base;
        int* start = v.data();
        int* end = v.data() + v.size();
        omp_set_num_threads(threads);
        auto t0 = std::chrono::high_resolution_clock::now();
        SMTimeMetrics m = run_sample_merge_dispatch(start, end, threads);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!std::is_sorted(v.begin(), v.end())) {
            std::cerr << "FAIL: sampleMerge no ordeno (threads=" << threads << ")\n";
            return 1;
        }

        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto s_to_ms = [](double s) { return s * 1000.0; };
        std::cout << "SampleMerge(" << threads << "): " << std::fixed << std::setprecision(3) << wall_ms << " ms"
                  << " | sort=" << s_to_ms(m.sortTime) << " pivot=" << s_to_ms(m.pivotTime)
                  << " class=" << s_to_ms(m.classificationTime) << " bucket=" << s_to_ms(m.bucketTime)
                  << " merge=" << s_to_ms(m.mergeTime)
                  << " copyback=" << s_to_ms(m.copybackTime) << " ms\n";
    }

    return 0;
}