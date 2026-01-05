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

template <unsigned int THREADS, typename T>
void simd_k_way_merge(AlignedBuckets<THREADS, T>& buckets, const std::array<ClassifiedRun<T>, THREADS * THREADS>& runs, T* base_ptr) {
    
    // 1. CONFIGURACIÓN DE TAGS
    // Usamos FixedTag con 'THREADS' explícito.
    // Si THREADS > Lanes nativos (ej. 64 threads en AVX2), Highway/Compilador
    // usarán múltiples registros físicos automáticamente (Register Spilling / Unrolling).
    using D_K      = hn::FixedTag<T, THREADS>;
    using D_Offset = hn::RebindToSigned<D_K>;
    using TI       = hn::TFromD<D_Offset>;
    using D_Native = hn::ScalableTag<T>; // Para la salida (siempre ancho nativo)

    const D_K dk;
    const D_Offset doff;
    const D_Native dn;
    const size_t Lanes = hn::Lanes(dn);

    // Nota: Eliminamos el static_assert que limitaba THREADS.
    // Solo ten en cuenta que un número muy alto (ej. > 256) podría aumentar 
    // el tiempo de compilación o presionar demasiado los registros.

    #pragma omp parallel for schedule(static)
    for (unsigned int bucketId = 0; bucketId < THREADS; ++bucketId) {
        
        // 2. SETUP
        alignas(HWY_ALIGNMENT) TI head_offsets[THREADS];
        alignas(HWY_ALIGNMENT) TI endm1_offsets[THREADS];
        alignas(HWY_ALIGNMENT) TI end_offsets[THREADS];
        
        size_t total_size = 0;
        
        for (unsigned int i = 0; i < THREADS; ++i) {
            const auto& run = runs[i * THREADS + bucketId];
            head_offsets[i]  = static_cast<TI>(run.start - base_ptr);
            const auto end   = static_cast<TI>(head_offsets[i] + static_cast<TI>(run.size));
            end_offsets[i]   = end;
            // Clamping limit: (end - 1). Si vacío, apuntamos a start (seguro).
            endm1_offsets[i] = (run.size > 0) ? (end - 1) : head_offsets[i];
            total_size += run.size;
        }

        // Cargamos el estado inicial en registros (o múltiples registros si THREADS es grande)
        auto v_offsets = hn::Load(doff, head_offsets);
        const auto v_end = hn::Load(doff, end_offsets);
        const auto v_endm1 = hn::Load(doff, endm1_offsets);
        
        T* out = buckets.ptrs[bucketId];
        
        // Padding para vectorización de salida
        size_t remainder = total_size & (Lanes - 1);
        T* end_aligned = out + (total_size - remainder); 

        const auto v_step = hn::Set(doff, 1);
        const auto v_zero = hn::Zero(doff);
        const auto v_iota = hn::Iota(doff, 0); 
        const auto v_sentinel = hn::Set(dk, std::numeric_limits<T>::max());

        // 3. HOT-LOOP (Unrolled por compilador gracias a FixedTag)
        while (out < end_aligned) {
            auto v_out = hn::Undefined(dn);

            // Bucle interno sobre el ancho NATIVO de salida
            for (size_t i = 0; i < Lanes; ++i) {
                
                // A. MÁSCARA DE ACTIVIDAD (Seguridad Lógica)
                // Evita condiciones de carrera y actualiza correctamente el estado.
                const auto active_off = hn::Lt(v_offsets, v_end);
                const auto active = hn::RebindMask(dk, active_off);

                // B. CLAMPING EXPLÍCITO (Velocidad TLB/Caché)
                // Forzamos que los índices siempre apunten a memoria válida (<= end-1).
                // Esto previene stalls por fallos de página especulativos en carriles inactivos.
                const auto v_clamped = hn::Min(v_offsets, v_endm1);
                
                // C. GATHER ENMASCARADO (Seguridad Memoria)
                // Lee datos válidos. Los inactivos retornan Sentinel (MAX).
                const auto v_vals = hn::MaskedGatherIndexOr(v_sentinel, active, dk, base_ptr, v_clamped);

                // D. REDUCCIÓN VERTICAL
                // Encontrar el mínimo entre los THREADS candidatos.
                auto v_min = hn::Set(dk, hn::ReduceMin(dk, v_vals));
                
                // E. SELECCIÓN DEL GANADOR
                auto mask = hn::Eq(v_vals, v_min);
                mask = hn::And(mask, active); // Desempate con sentinelas
                
                const intptr_t winner_idx = hn::FindFirstTrue(dk, mask);
                
                // Generar máscara para incrementar solo el offset ganador
                const auto winner_mask_off = hn::Eq(v_iota, hn::Set(doff, static_cast<TI>(winner_idx)));

                // F. UPDATE & INSERT
                v_offsets = hn::Add(v_offsets, hn::IfThenElse(winner_mask_off, v_step, v_zero));
                
                // Insertamos en el vector de salida (que es de tamaño nativo 'Lanes')
                v_out = hn::InsertLane(v_out, i, hn::GetLane(v_min));
            }
            
            // Escribir bloque de salida
            hn::Stream(v_out, dn, out);
            out += Lanes;
        }

        // 4. CLEANUP (Misma lógica para el resto)
        if (remainder > 0) {
            auto v_out = hn::Undefined(dn);
            for (size_t i = 0; i < remainder; ++i) {
                const auto active_off = hn::Lt(v_offsets, v_end);
                const auto active = hn::RebindMask(dk, active_off);
                const auto v_clamped = hn::Min(v_offsets, v_endm1);
                
                const auto v_vals = hn::MaskedGatherIndexOr(v_sentinel, active, dk, base_ptr, v_clamped);
                auto v_min = hn::Set(dk, hn::ReduceMin(dk, v_vals));
                
                auto mask = hn::Eq(v_vals, v_min);
                mask = hn::And(mask, active);
                
                const intptr_t winner_idx = hn::FindFirstTrue(dk, mask);
                const auto winner_mask_off = hn::Eq(v_iota, hn::Set(doff, static_cast<TI>(winner_idx)));
                
                v_offsets = hn::Add(v_offsets, hn::IfThenElse(winner_mask_off, v_step, v_zero));
                v_out = hn::InsertLane(v_out, i, hn::GetLane(v_min));
            }
            hn::StoreN(v_out, dn, out, remainder);
        }
    }
}
} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();


HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <unsigned int THREADS, typename T>
void grid_k_way_merge(T* out_ptr, const std::array<ClassifiedRun<T>, THREADS * THREADS>& runs, unsigned int bucketId, T* base_ptr) {
    
    // 1. CONFIGURACIÓN SIMD
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    const size_t Lanes = hn::Lanes(d);
    
    // Calculamos vectores necesarios (Techo de la división)
    constexpr size_t NUM_VECS = (THREADS + hn::Lanes(hn::ScalableTag<T>()) - 1) / hn::Lanes(hn::ScalableTag<T>());

    // SAFETY PAD: Dimensionamos los arrays al tamaño FÍSICO de los registros, no al lógico.
    // Si THREADS=6 y Lanes=8, creamos arrays de 8. Esto evita corrupción de memoria si se acceden los lanes fantasma.
    constexpr size_t PADDED_SIZE = NUM_VECS * hn::Lanes(hn::ScalableTag<T>());

    // 2. ESTADO EN MEMORIA (STACK PADDED)
    // Inicializamos todo a valores seguros/nulos para evitar sorpresas en lanes fantasma.
    const T* cursors[PADDED_SIZE];
    const T* ends[PADDED_SIZE];
    
    // Llenar con datos reales
    size_t total_elements = 0;
    for (unsigned int i = 0; i < THREADS; ++i) {
        const auto& run = runs[i * THREADS + bucketId];
        cursors[i] = run.start;
        ends[i] = run.start + run.size;
        total_elements += run.size;
    }
    
    // Rellenar lanes fantasma (si THREADS < PADDED_SIZE) con punteros seguros
    // Aunque la lógica no debería tocarlos, esto previene segfaults en lecturas especulativas.
    for (unsigned int i = THREADS; i < PADDED_SIZE; ++i) {
        cursors[i] = nullptr; 
        ends[i] = nullptr;
    }

    // 3. ESTADO EN REGISTROS (GRID)
    V current_vals[NUM_VECS];
    V current_idxs[NUM_VECS];
    const auto v_max = hn::Set(d, std::numeric_limits<T>::max());

    for (size_t v = 0; v < NUM_VECS; ++v) {
        current_vals[v] = v_max;
        current_idxs[v] = hn::Iota(d, v * Lanes); // IDs: 0..7, 8..15...
        
        for (size_t l = 0; l < Lanes; ++l) {
            size_t src_idx = v * Lanes + l;
            // Solo cargamos si es un thread válido y tiene datos
            if (src_idx < THREADS && cursors[src_idx] < ends[src_idx]) {
                current_vals[v] = hn::InsertLane(current_vals[v], l, *cursors[src_idx]);
            }
        }
    }

    // 4. BUFFER DE SALIDA (Registro Acumulador)
    V v_out_buffer = hn::Undefined(d);
    size_t out_lane_idx = 0;

    // 5. HOT LOOP
    while (total_elements > 0) {
        
        // --- A. VERTICAL MIN ---
        V min_val = current_vals[0];
        V min_idx = current_idxs[0]; 

        // Unroll automático
        for (size_t v = 1; v < NUM_VECS; ++v) {
            const auto mask = hn::Lt(current_vals[v], min_val);
            min_val = hn::Min(current_vals[v], min_val);
            min_idx = hn::IfThenElse(mask, current_idxs[v], min_idx);
        }

        // --- B. REDUCCIÓN Y BUFFERING ---
        T scalar_min = hn::ReduceMin(d, min_val);
        
        // Insertamos en el registro buffer
        v_out_buffer = hn::InsertLane(v_out_buffer, out_lane_idx, scalar_min);
        out_lane_idx++;

        // --- STREAMING (CRÍTICO) ---
        // Si el buffer se llena, vertemos a RAM usando Non-Temporal Store.
        if (out_lane_idx == Lanes) {
            // ASUMIMOS que out_ptr está alineado gracias a tu lógica de buckets.
            // Si buckets.ptrs[bucketId] no es múltiplo de alignof(Vec), esto fallará.
            hn::Stream(v_out_buffer, d, out_ptr);
            out_ptr += Lanes;
            out_lane_idx = 0;
        }
        
        total_elements--;

        // --- C. IDENTIFICAR ORIGEN ---
        auto mask_winner = hn::Eq(min_val, hn::Set(d, scalar_min));
        int lane_winner = hn::FindFirstTrue(d, mask_winner);
        
        // Recuperamos el índice REAL de la fuente (0..THREADS-1)
        int source_idx = static_cast<int>(hn::ExtractLane(min_idx, lane_winner));

        // --- D. UPDATE (SAFE & FAST) ---
        // Aquí estaba el riesgo con 6 threads. Ahora cursors tiene tamaño PADDED_SIZE,
        // así que incluso si source_idx es 6 o 7 (fantasma), no corrompemos memoria ajena.
        
        cursors[source_idx]++;
        
        // Branchless check
        // Si es fantasma, ends[source_idx] es nullptr, la comparación falla segura.
        bool has_next = cursors[source_idx] < ends[source_idx];
        T next_val = has_next ? *cursors[source_idx] : std::numeric_limits<T>::max();

        // Cálculo optimizado por compilador (shifts)
        size_t target_vec = source_idx / Lanes;
        size_t target_lane = source_idx % Lanes;
        
        // Inserción quirúrgica
        current_vals[target_vec] = hn::InsertLane(current_vals[target_vec], target_lane, next_val);
    }

    // 6. FLUSH DEL REMANENTE
    // Los últimos elementos no llenan un vector, usamos Store parcial seguro.
    if (out_lane_idx > 0) {
        hn::StoreN(v_out_buffer, d, out_ptr, out_lane_idx);
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

template <typename T>
void sortBlocks(T* start, T* end, size_t blockSize) {
    #pragma omp parallel for schedule(static)
    for (size_t offset = 0; offset < (size_t)(end - start); offset += blockSize) {
        T* blockStart = start + offset;
        T* blockEnd = std::min(blockStart + blockSize, end);
        if (blockEnd > blockStart) {
            hwy::VQSort(blockStart, blockEnd - blockStart, hwy::SortAscending());
        }
    }
}

template <unsigned int THREADS, typename T>
inline std::array<T, THREADS - 1> getPivots(T* start, T* end, size_t block_size) {
    std::array<T, THREADS - 1> pivots;
    const size_t step = std::max<size_t>(1, block_size / THREADS);

    #pragma omp parallel for schedule(static)
    for (unsigned int p = 0; p < THREADS - 1; ++p) {
        T candidates[THREADS]; 
        const size_t relative_idx = (p + 1) * step - 1;

        for (unsigned int b = 0; b < THREADS; ++b) {
            T* blockStart = start + b * block_size;
            T* blockEnd = std::min(blockStart + block_size, end);
            size_t currentSize = (blockEnd > blockStart) ? (blockEnd - blockStart) : 0;
            size_t idx = std::min(relative_idx, currentSize > 0 ? currentSize - 1 : 0);
            candidates[b] = blockStart[idx];
        }
        hwy::VQSort(candidates, THREADS, hwy::SortAscending());
        pivots[p] = candidates[THREADS / 2];
    }
    hwy::VQSort(pivots.data(), THREADS - 1, hwy::SortAscending());
    return pivots;
}

template<unsigned int THREADS, typename T>
std::array<ClassifiedRun<T>, THREADS*THREADS> classifyElements(T* start, T* end, size_t block_size, const std::array<T, THREADS-1> &pivots) {
    std::array<ClassifiedRun<T>, THREADS*THREADS> classifiedRuns;
    #pragma omp parallel for schedule(static)
    for (int blockId = 0; blockId < THREADS; ++blockId) {
        std::array<ClassifiedRun<T>, THREADS> localRuns;
        T* blockStart = start + blockId * block_size;
        T* blockEnd = std::min(blockStart + block_size, end);
        T* currentSearchStart = blockStart;

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
        // Padding con ALIGN dinámico
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
        return {
            .totalTime = std::chrono::duration<double>(blockDuration).count(),
            .sortTime = std::chrono::duration<double>(blockDuration).count(),
            .pivotTime = 0.0,
            .classificationTime = 0.0,
            .bucketTime = 0.0,
            .mergeTime = 0.0,
            .copybackTime = 0.0
        };
    }
    else if (THREADS == 2) {
        // Caso especial para 2 hilos: std::inplace_merge directo
        std::vector<T> temp(arrSize);

        auto blockTimer = std::chrono::high_resolution_clock::now();
        sortBlocks(start, end, arrSize / THREADS);
        auto blockDuration = std::chrono::high_resolution_clock::now() - blockTimer;

        auto mergeTimer = std::chrono::high_resolution_clock::now();
        std::merge(
            std::execution::seq,
            start, start + arrSize / 2,
            start + arrSize / 2, end,
            temp.data()
        );
        std::memcpy(start, temp.data(), arrSize * sizeof(T));
        std::thread([delTemp = std::move(temp)]() {}).detach();
        auto mergeDuration = std::chrono::high_resolution_clock::now() - mergeTimer;

        return {
            .totalTime = std::chrono::duration<double>(blockDuration + mergeDuration).count(),
            .sortTime = std::chrono::duration<double>(blockDuration).count(),
            .pivotTime = 0.0,
            .classificationTime = 0.0,
            .bucketTime = 0.0,
            .mergeTime = std::chrono::duration<double>(mergeDuration).count(),
            .copybackTime = 0.0
        };
    }

    size_t block_size = arrSize / THREADS;
    if (block_size == 0) block_size = 1;
    auto oldNumThreads = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    
    auto blockTimer = std::chrono::high_resolution_clock::now();
    sortBlocks(start, end, block_size);
    auto blockDuration = std::chrono::high_resolution_clock::now() - blockTimer;

    auto sampleTimer = std::chrono::high_resolution_clock::now();
    auto pivots = getPivots<THREADS>(start, end, block_size);
    auto sampleDuration = std::chrono::high_resolution_clock::now() - sampleTimer;

    auto classificationTimer = std::chrono::high_resolution_clock::now();
    auto classifiedRuns = classifyElements<THREADS>(start, end, block_size, pivots);
    auto classificationDuration = std::chrono::high_resolution_clock::now() - classificationTimer;

    auto bucketTimer = std::chrono::high_resolution_clock::now();
    auto buckets = createAlignedBuckets<THREADS, int>(classifiedRuns);
    auto bucketDuration = std::chrono::high_resolution_clock::now() - bucketTimer;

    auto mergeTimer = std::chrono::high_resolution_clock::now();
    parallel_grid_merge<THREADS>(buckets, classifiedRuns, start); ////hn::simd_k_way_merge_v2<THREADS>(buckets, classifiedRuns, start);
    auto mergeDuration = std::chrono::high_resolution_clock::now() - mergeTimer;

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