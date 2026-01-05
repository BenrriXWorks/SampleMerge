#ifndef SAMPLE_MERGE_HPP
#define SAMPLE_MERGE_HPP

#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <limits> 
#include <thread>
#include <execution>
#include <math.h>

#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"

// --------------------------------------------------------------------------
// ESTRUCTURAS GLOBALES
// --------------------------------------------------------------------------

template<typename T>
struct ClassifiedRun {
    T* start;
    size_t size;
};

template<unsigned int THREADS, typename T>
struct AlignedBuckets {
    T* base;
    std::array<T*, THREADS> ptrs;
    std::array<size_t, THREADS> sizes_unpadded;
};

// --------------------------------------------------------------------------
// CÓDIGO SIMD
// --------------------------------------------------------------------------
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// --------------------------------------------------------------------------
// GRID K-WAY MERGE V6 con prefetch
// --------------------------------------------------------------------------
template <unsigned int THREADS, typename T>
void grid_k_way_merge(T* out_ptr, const std::array<ClassifiedRun<T>, THREADS * THREADS>& runs, unsigned int bucketId, T* base_ptr) {
    
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    const size_t Lanes = hn::Lanes(d);
    
    constexpr size_t NUM_VECS = (THREADS + hn::Lanes(hn::ScalableTag<T>()) - 1) / hn::Lanes(hn::ScalableTag<T>());
    constexpr size_t PADDED_SIZE = NUM_VECS * hn::Lanes(hn::ScalableTag<T>());

    // --- SETUP MEMORIA ---
    const T* cursors[PADDED_SIZE];
    const T* ends[PADDED_SIZE];
    size_t total_elements = 0;
    
    for (unsigned int i = 0; i < THREADS; ++i) {
        const auto& run = runs[i * THREADS + bucketId];
        cursors[i] = run.start;
        ends[i] = run.start + run.size;
        total_elements += run.size;
    }
    for (unsigned int i = THREADS; i < PADDED_SIZE; ++i) {
        cursors[i] = nullptr; ends[i] = nullptr;
    }

    // --- SETUP REGISTROS ---
    V current_vals[NUM_VECS];
    V current_idxs[NUM_VECS];
    const auto v_max = hn::Set(d, std::numeric_limits<T>::max());

    for (size_t v = 0; v < NUM_VECS; ++v) {
        current_vals[v] = v_max;
        current_idxs[v] = hn::Iota(d, v * Lanes);
        
        for (size_t l = 0; l < Lanes; ++l) {
            size_t src_idx = v * Lanes + l;
            if (src_idx < THREADS && cursors[src_idx] < ends[src_idx]) {
                current_vals[v] = hn::InsertLane(current_vals[v], l, *cursors[src_idx]);
            }
        }
    }

    V v_out_buffer = hn::Undefined(d);
    
    size_t full_vectors = total_elements / Lanes;
    size_t remaining = total_elements % Lanes;

    // --- HOT LOOP ---
    while (full_vectors > 0) {
        
        for (size_t k = 0; k < Lanes; ++k) {
            
            // 1. Min Vertical
            V min_val = current_vals[0];
            V min_idx = current_idxs[0]; 

            for (size_t v = 1; v < NUM_VECS; ++v) {
                const auto mask = hn::Lt(current_vals[v], min_val);
                min_val = hn::Min(current_vals[v], min_val);
                min_idx = hn::IfThenElse(mask, current_idxs[v], min_idx);
            }

            // 2. Reduce & Buffer
            T scalar_min = hn::ReduceMin(d, min_val);
            v_out_buffer = hn::InsertLane(v_out_buffer, k, scalar_min);

            // 3. Identificar Origen
            auto mask_winner = hn::Eq(min_val, hn::Set(d, scalar_min));
            int lane_winner = hn::FindFirstTrue(d, mask_winner);
            int source_idx = static_cast<int>(hn::ExtractLane(min_idx, lane_winner));

            // 4. Update & PREFETCH
            cursors[source_idx]++;
            
            
            // PREFETCH MANUAL:
            // Solicitamos a la cache que traiga el dato futuro.
            // +2 o +4 suele ser mejor que +1 para romper la latencia de banco.
            // __builtin_prefetch(addr, rw, locality): rw=0(read), locality=3(high temporal)
            if (HWY_LIKELY(source_idx < THREADS)) { // Check estático para compilador
                //hwy::PrefetchT0(cursors[source_idx] + 4); 
                __builtin_prefetch(cursors[source_idx] + 4, 0, 3);
            }
            

            T next_val;
            if (HWY_LIKELY(cursors[source_idx] < ends[source_idx])) {
                next_val = *cursors[source_idx];
            } else {
                next_val = std::numeric_limits<T>::max();
            }

            size_t target_vec = source_idx / Lanes;
            size_t target_lane = source_idx % Lanes;
            current_vals[target_vec] = hn::InsertLane(current_vals[target_vec], target_lane, next_val);
        }

        hn::Stream(v_out_buffer, d, out_ptr);
        out_ptr += Lanes;
        full_vectors--;
    }

    // --- CLEANUP ---
    size_t out_idx = 0;
    while (remaining > 0) {
        V min_val = current_vals[0];
        V min_idx = current_idxs[0]; 
        for (size_t v = 1; v < NUM_VECS; ++v) {
            auto mask = hn::Lt(current_vals[v], min_val);
            min_val = hn::Min(current_vals[v], min_val);
            min_idx = hn::IfThenElse(mask, current_idxs[v], min_idx);
        }

        T scalar_min = hn::ReduceMin(d, min_val);
        v_out_buffer = hn::InsertLane(v_out_buffer, out_idx++, scalar_min);

        auto mask_winner = hn::Eq(min_val, hn::Set(d, scalar_min));
        int lane_winner = hn::FindFirstTrue(d, mask_winner);
        int source_idx = static_cast<int>(hn::ExtractLane(min_idx, lane_winner));

        cursors[source_idx]++;
        T next_val = (cursors[source_idx] < ends[source_idx]) ? *cursors[source_idx] : std::numeric_limits<T>::max();
        
        size_t target_vec = source_idx / Lanes;
        size_t target_lane = source_idx % Lanes;
        current_vals[target_vec] = hn::InsertLane(current_vals[target_vec], target_lane, next_val);
        
        remaining--;
    }
    
    if (out_idx > 0) {
        hn::StoreN(v_out_buffer, d, out_ptr, out_idx);
    }
}

} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();


template <unsigned int THREADS, typename T>
void parallel_grid_merge(AlignedBuckets<THREADS, T>& buckets,
                         const std::array<ClassifiedRun<T>, THREADS * THREADS>& runs,
                         T* base_ptr) {             
    namespace hn = hwy::HWY_NAMESPACE;
    #pragma omp parallel for schedule(static)
    for (unsigned int bucketId = 0; bucketId < THREADS; ++bucketId) {
        hn::grid_k_way_merge<THREADS, T>(buckets.ptrs[bucketId], runs, bucketId, base_ptr);
    }
}

// --------------------------------------------------------------------------
// CÓDIGO HOST
// --------------------------------------------------------------------------

namespace hn = hwy::HWY_NAMESPACE;

template <unsigned int THREADS, typename T>
void sortBlocksExact(T* start, size_t total_n) {
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < THREADS; ++i) {
        size_t start_idx = (size_t(i) * total_n) / THREADS;
        size_t end_idx   = (size_t(i + 1) * total_n) / THREADS;
        
        if (end_idx > start_idx) {
            hwy::VQSort(start + start_idx, end_idx - start_idx, hwy::SortAscending());
        }
    }
}

template <unsigned int THREADS, typename T>
inline std::array<T, THREADS - 1> getPivotsExact(T* start, size_t total_n) {
    std::array<T, THREADS - 1> pivots;

    #pragma omp parallel for schedule(static)
    for (unsigned int p = 0; p < THREADS - 1; ++p) {
        T candidates[THREADS]; 
        
        for (unsigned int b = 0; b < THREADS; ++b) {
            // Recalcular los límites del bloque 'b' exactamente igual que en sort
            size_t b_start_idx = (size_t(b) * total_n) / THREADS;
            size_t b_end_idx   = (size_t(b + 1) * total_n) / THREADS;
            size_t currentSize = b_end_idx - b_start_idx;

            T* blockStart = start + b_start_idx;
            
            // Step relativo al tamaño real de este bloque
            size_t step = std::max<size_t>(1, currentSize / THREADS);
            size_t idx = std::min((p + 1) * step - 1, currentSize > 0 ? currentSize - 1 : 0);
            
            candidates[b] = blockStart[idx];
        }
        hwy::VQSort(candidates, THREADS, hwy::SortAscending());
        pivots[p] = candidates[THREADS / 2];
    }
    hwy::VQSort(pivots.data(), THREADS - 1, hwy::SortAscending());
    return pivots;
}

template<unsigned int THREADS, typename T>
std::array<ClassifiedRun<T>, THREADS*THREADS> classifyElementsExact(T* start, size_t total_n, const std::array<T, THREADS-1> &pivots) {
    std::array<ClassifiedRun<T>, THREADS*THREADS> classifiedRuns;
    
    #pragma omp parallel for schedule(static)
    for (int blockId = 0; blockId < THREADS; ++blockId) {
        // Partición exacta
        size_t start_idx = (size_t(blockId) * total_n) / THREADS;
        size_t end_idx   = (size_t(blockId + 1) * total_n) / THREADS;
        
        T* blockStart = start + start_idx;
        T* blockEnd   = start + end_idx; // Esto asegura que el último bloque llegue al final real
        T* currentSearchStart = blockStart;

        std::array<ClassifiedRun<T>, THREADS> localRuns;

        for (int pivotId = 0; pivotId < THREADS - 1; ++pivotId) {
            T* pos = std::lower_bound(currentSearchStart, blockEnd, pivots[pivotId]);
            localRuns[pivotId] = {currentSearchStart, static_cast<size_t>(pos - currentSearchStart)};
            currentSearchStart = pos; 
        }
        localRuns[THREADS - 1] = {currentSearchStart, static_cast<size_t>(blockEnd - currentSearchStart)};
        
        std::memcpy(classifiedRuns.data() + blockId * THREADS, localRuns.data(), sizeof(ClassifiedRun<T>) * THREADS);
    }
    return classifiedRuns;
}

template<unsigned int THREADS, typename T, size_t ALIGN = HWY_ALIGNMENT> 
AlignedBuckets<THREADS, T> createAlignedBuckets(const std::array<ClassifiedRun<T>, THREADS*THREADS>& runs) {

    constexpr size_t SENTINEL_ELEMENTS = HWY_MAX_BYTES / sizeof(T); 
    
    std::array<size_t, THREADS> sizes = {0};
    for (unsigned int i = 0; i < THREADS * THREADS; ++i) {
        sizes[i % THREADS] += runs[i].size;
    }
    
    size_t currentOffset = 0;
    std::array<size_t, THREADS> offsets;
    
    for (unsigned int b = 0; b < THREADS; ++b) {
        offsets[b] = currentOffset;
        size_t dataBytes = sizes[b] * sizeof(T);
        size_t sentinelBytes = SENTINEL_ELEMENTS * sizeof(T);
        
        size_t totalBytes = dataBytes + sentinelBytes;
        size_t paddedBytes = (totalBytes + ALIGN - 1) & ~(ALIGN - 1);
        currentOffset += paddedBytes / sizeof(T);
    }

    AlignedBuckets<THREADS, T> result;
    result.base = static_cast<T*>(hwy::AllocateAlignedBytes(currentOffset * sizeof(T)));
    
    for (unsigned int b = 0; b < THREADS; ++b) {
        result.ptrs[b] = result.base + offsets[b];
        result.sizes_unpadded[b] = sizes[b];
        T* endOfData = result.ptrs[b] + sizes[b];
        std::fill(endOfData, endOfData + SENTINEL_ELEMENTS, std::numeric_limits<T>::max());
    }
    return result;
}

template <unsigned int THREADS, typename T>
void copyback_buckets(AlignedBuckets<THREADS, T>& buckets, T* dest) {
    std::array<size_t, THREADS> write_offsets;
    size_t current_pos = 0;
    for (unsigned int b = 0; b < THREADS; ++b) {
        write_offsets[b] = current_pos;
        current_pos += buckets.sizes_unpadded[b];
    }
    #pragma omp parallel for schedule(static)
    for (unsigned int b = 0; b < THREADS; ++b) {
        std::memcpy(dest + write_offsets[b], buckets.ptrs[b], buckets.sizes_unpadded[b] * sizeof(T));
    }
}

struct SMTimeMetrics {
    double totalTime;     
    double sortTime;      
    double pivotTime;     
    double classificationTime; 
    double bucketTime;    
    double mergeTime;     
    double copybackTime;  
};

template <int THREADS, typename T>
SMTimeMetrics sampleMerge(T* start, T* end) {
    size_t arrSize = end - start;
    if (arrSize == 0) return {};
    else if (THREADS == 1) {
        auto blockTimer = std::chrono::high_resolution_clock::now();
        VQSort(start, arrSize, hwy::SortAscending());
        auto blockDuration = std::chrono::high_resolution_clock::now() - blockTimer;
        return {std::chrono::duration<double>(blockDuration).count(), std::chrono::duration<double>(blockDuration).count(), 0,0,0,0,0};
    } else if (THREADS == 2) {
        auto blockTimer = std::chrono::high_resolution_clock::now();
        std::array<std::thread, 2> sortThreads {std::thread([start, arrSize]() {
            hwy::VQSort(start, arrSize / 2, hwy::SortAscending());
        }), std::thread([start, arrSize]() {
            hwy::VQSort(start + arrSize / 2, arrSize - arrSize / 2, hwy::SortAscending());
        })};
        for (auto& thread : sortThreads) thread.join();
        auto blockDuration = std::chrono::high_resolution_clock::now() - blockTimer;
        
        auto mergeTimer = std::chrono::high_resolution_clock::now();
        std::vector<T> temp(arrSize);
        std::merge(
            std::execution::unseq,
            start, start + arrSize / 2,
            start + arrSize / 2, end,
            temp.data()
        );
        auto mergeDuration = std::chrono::high_resolution_clock::now() - mergeTimer;
        auto copybackTimer = std::chrono::high_resolution_clock::now();
        std::memcpy(start, temp.data(), arrSize * sizeof(T));
        auto copybackDuration = std::chrono::high_resolution_clock::now() - copybackTimer;
        std::thread([delTemp = std::move(temp)]() {}).detach();
        return {
            .totalTime = std::chrono::duration<double>(blockDuration + mergeDuration).count(),
            .sortTime = std::chrono::duration<double>(blockDuration).count(),
            .pivotTime = 0.0,
            .classificationTime = 0.0,
            .bucketTime = 0.0,
            .mergeTime = std::chrono::duration<double>(mergeDuration).count(),
            .copybackTime = std::chrono::duration<double>(copybackDuration).count()
        };
    }

    auto oldNumThreads = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    
    // 1. SORT BLOCKS (Exact Partitioning)
    auto blockTimer = std::chrono::high_resolution_clock::now();
    sortBlocksExact<THREADS>(start, arrSize);
    auto blockDuration = std::chrono::high_resolution_clock::now() - blockTimer;

    // 2. PIVOTS (Exact Partitioning)
    auto sampleTimer = std::chrono::high_resolution_clock::now();
    auto pivots = getPivotsExact<THREADS>(start, arrSize);
    auto sampleDuration = std::chrono::high_resolution_clock::now() - sampleTimer;

    // 3. CLASSIFY (Exact Partitioning)
    auto classificationTimer = std::chrono::high_resolution_clock::now();
    auto classifiedRuns = classifyElementsExact<THREADS>(start, arrSize, pivots);
    auto classificationDuration = std::chrono::high_resolution_clock::now() - classificationTimer;

    // 4. BUCKETS
    auto bucketTimer = std::chrono::high_resolution_clock::now();
    auto buckets = createAlignedBuckets<THREADS, int>(classifiedRuns);
    auto bucketDuration = std::chrono::high_resolution_clock::now() - bucketTimer;

    // 5. MERGE (SIMD Grid)
    auto mergeTimer = std::chrono::high_resolution_clock::now();
    parallel_grid_merge<THREADS>(buckets, classifiedRuns, start); 
    auto mergeDuration = std::chrono::high_resolution_clock::now() - mergeTimer;

    // 6. COPYBACK
    auto copybackTimer = std::chrono::high_resolution_clock::now();
    copyback_buckets<THREADS>(buckets, start);
    auto copybackDuration = std::chrono::high_resolution_clock::now() - copybackTimer;

    // Cleanup
    std::thread([bucketsPtr = buckets.base]() {
        hwy::FreeAlignedBytes(bucketsPtr, nullptr, nullptr);
    }).detach();

    omp_set_num_threads(oldNumThreads);

    return {
        .totalTime = std::chrono::duration<double>(blockDuration + sampleDuration + classificationDuration + bucketDuration + mergeDuration + copybackDuration).count(),
        .sortTime = std::chrono::duration<double>(blockDuration).count(),
        .pivotTime = std::chrono::duration<double>(sampleDuration).count(),
        .classificationTime = std::chrono::duration<double>(classificationDuration).count(),
        .bucketTime = std::chrono::duration<double>(bucketDuration).count(),
        .mergeTime = std::chrono::duration<double>(mergeDuration).count(),
        .copybackTime = std::chrono::duration<double>(copybackDuration).count()
    };
}


#endif // SAMPLE_MERGE_HPP
