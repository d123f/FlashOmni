#ifndef SPARSEQKV_GEMM_CUH_
#define SPARSEQKV_GEMM_CUH_

#include <cstdint>
#include <optional>
#include <sstream>

// #include <cuda.h>
// #include <cublas_v2.h>
#include <stdlib.h>
#include <sys/types.h>

#include <type_traits>
#include <queue>  
#include <vector>

#include "cute/algorithm/clear.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/fast_math.h"
#include "../../utils.cuh"
#include "../../cutlass_utils.cuh"
#include "cutlass/array.h"
#include "cutlass/half.h"

namespace sparseqkv {

namespace gemm {
using namespace cute;
using namespace cutlass;

template <int lut> 
__device__ static int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(res)
                 : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
  }

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ static auto dequant_fp16(uint32_t i8s) {
    using result_type = cutlass::Array<cutlass::half_t, 4>;
    result_type result;
    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    static constexpr uint32_t mask_for_elt_01     = 0x5250;
    static constexpr uint32_t mask_for_elt_23     = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));


    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

    return result;
}

__device__ static auto dequant_bf16(uint32_t i8s) {
    using result_type = cutlass::Array<bfloat16_t, 4>;
    result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    uint32_t*      bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);

    static constexpr uint32_t fp32_base = 0x4B000000;
    float                     fp32_intermediates[4];

    // Construct FP32s, bfloat does not have enough mantissa for IADD trick
    uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
    fp32_intermediates_casted[0]        = __byte_perm(i8s, fp32_base, 0x7650);
    fp32_intermediates_casted[1]        = __byte_perm(i8s, fp32_base, 0x7652);
    fp32_intermediates_casted[2]        = __byte_perm(i8s, fp32_base, 0x7651);
    fp32_intermediates_casted[3]        = __byte_perm(i8s, fp32_base, 0x7653);

    // Subtract out fp32_base + 128 to make the unsigned integer signed.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < 4; ++ii) {
        fp32_intermediates[ii] -= 8388736.f;
    }

    // Truncate the fp32 representation and pack up as bfloat16s.
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < 2; ++ii) {
        bf16_result_ptr[ii] =
            __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
    }
#else
    // Disable this on architectures older than Ampere since they lack hardware for bf16 mma. If one wishes to use
    // HMMA on older hardware, they should Convert directly to FP16 using FP16 converters.
    result.clear();  // Suppress compiler warning
    cutlass::arch::device_breakpoint();
#endif
    return result;
}


template <typename T, typename Config>
__global__ void sparseqkv_gemm_reduction_quant_kernel(const T *Aptr, const uint8_t *Bptr,  T *bias_ptr,
                         T *Dptr, int m, int n, int k,
                        uint32_t num_qo_heads, uint32_t head_dim,
                        uint8_t* sparse_info, uint32_t* sparse_info_indptr,
                        uint32_t num_text_tokens,  bool is_for_cached) {
    
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;


    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kTileK_B_I8 = Config::kTileK_B_I8;
    constexpr int kStage = Config::kStage;
    // Initilize shared memory
    extern __shared__ char smem_buffer[]; // Dynamically sized shared memory
    T* Ashm = reinterpret_cast<T*>(smem_buffer);
    T* Bshm = reinterpret_cast<T*>(Ashm + cute::cosize(SmemLayoutA{}));
    
    // Initilize thread block
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;
    const T *Aptr_cur = Aptr + iz * m * k;
    const T *Bptr_cur = reinterpret_cast<const T*>(Bptr);
    T *Dptr_cur = Dptr + iz * m * n;
    
    uint8_t* sparse_info_cur = sparse_info + sparse_info_indptr[iz];
    const uint32_t tile_per_head = head_dim / kTileK;
    // // B, N, H

    Tensor A = make_tensor(make_gmem_ptr(Aptr_cur), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr_cur), make_shape(n, k / 2), make_stride(k / 2, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr_cur), make_shape(m, n), make_stride(n, Int<1>{}));

    // slice the tensor to small one which is used for current thread block.
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _)); // (kTileM, kTileK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK_B_I8>{}), make_coord(ix, _)); // (kTileN, kTileK / 2, num_tile_k)
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix)); // (kTileM, kTileN) 

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (kTileN, kTileK / 2, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // auto tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,kStage)
    // auto tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,kStage)
    // auto tCgD = thr_mma.partition_C(gD);                                // (MMA, MMA_M, MMA_N)

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K / 2)
    auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)
    if (!is_for_cached) {
        const T* current_cta_bias_ptr = bias_ptr + iz * m * n;
        Tensor Bias = make_tensor(make_gmem_ptr(current_cta_bias_ptr), make_shape(m, n), make_stride(n, Int<1>{}));
        Tensor gBias = local_tile(Bias, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix)); // (kTileM, kTileN) 
        auto gmem_bias_src_partition = thr_mma.partition_C(gBias);
        copy(gmem_bias_src_partition, tCrD);
    } else if (bias_ptr != nullptr) {
        const T* current_cta_bias_ptr = bias_ptr + ix * kTileN;
        Tensor gBias = make_tensor(
            make_gmem_ptr(current_cta_bias_ptr), shape(gD), make_stride(Int<0>{}, Int<1>{})
        );
        auto gmem_bias_src_partition = thr_mma.partition_C(gBias);
        copy(gmem_bias_src_partition, tCrD);
    } else{
        clear(tCrD);
    }

    // from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, num_tile_k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, num_tile_k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K / 2, kStage)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K / 2)
    auto tCrB_dequant = make_tensor_like(tCrB); // (MMA, MMA_M, 1)
    /* PREFETCH */
    // submit kStage - 1 tile
    // gmem -> shm
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    // sparseqkv for gemm who is after attn
    uint32_t sparse_info_offset, sparse_info_offset_loaded = (uint32_t)-1;
    const uint32_t sparse_kv_info_offset_base = (num_text_tokens + iy * kTileM) / 128 * num_qo_heads;
    bool execute_bools[8];
    uint32_t cur_cached_stages = 0;
    // loop over k: i. load tile, ii. mma
    int ntile = (k + kTileK - 1) / kTileK;
// #pragma unroll
    for (;itile_to_read < ntile && cur_cached_stages < kStage - 1;) {
        sparse_info_offset = sparse_kv_info_offset_base + itile_to_read * kTileK / head_dim;;
        uint32_t sparse_info_offset_mod_8 = sparse_info_offset % 8;
        if (sparse_info_offset / 8 != sparse_info_offset_loaded) {
            sparse_info_offset_loaded = sparse_info_offset / 8;
            uint8_t sparse_info_current_value = sparse_info_cur[sparse_info_offset_loaded];
            for (int mod_id = sparse_info_offset_mod_8; mod_id < 8; ++mod_id) {
                execute_bools[mod_id] = (sparse_info_current_value >> mod_id) & 1;
            }
        }
        bool execute_this_tile = is_for_cached ? !execute_bools[sparse_info_offset_mod_8] : execute_bools[sparse_info_offset_mod_8];
        // bool execute_this_tile = true;
        if (execute_this_tile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                    tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                        tBsB_copy(_, _, _, ismem_write));
            cp_async_fence();
            ++ismem_write;
            ++cur_cached_stages;
            ++itile_to_read;
        } else {
            // itile_to_read += tile_per_head - (itile_to_read % tile_per_head);
            itile_to_read += tile_per_head;
        }
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik)); // tAsA: (CPY, CPY_M, CPY_K, kStage) tCrA_view: (CPY, CPY_M, CPY_K)
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik)); // tBsB: (CPY, CPY_N, CPY_K, kStage) tCrB_view: (CPY, CPY_N, CPY_K)

    // #pragma unroll 1
    // for (int itile = 0; itile < ntile; ++itile) {
    while (cur_cached_stages != 0) {
        int nk = size<2>(tCrA); // (MMA, MMA_M, MMA_K)
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            
            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                cur_cached_stages--;
                ismem_read = (ismem_read + 1) % kStage;
            }
            
            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), // tAsA: (CPY, CPY_M, CPY_K, kStage)
                    tCrA_view(_, _, ik_next));                            // tCrA_view: (CPY, CPY_M, CPY_K)
            
            auto tCrB_quant = tCrB(make_coord(_, ik % 2), _, ik / 2); // (MMA, MMA_M, 1)
            
            for (int de_m = 0; de_m < size<1>(tCrB_quant); ++de_m) {
                uint32_t f = *reinterpret_cast<uint32_t*>(&tCrB_quant(0, de_m));
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    auto result = dequant_bf16(f);
                    tCrB_dequant(0, de_m, ik / 2) = result[0];
                    tCrB_dequant(2, de_m, ik / 2) = result[1];
                    tCrB_dequant(1, de_m, ik / 2) = result[2];
                    tCrB_dequant(3, de_m, ik / 2) = result[3];
                } else if constexpr (std::is_same_v<T, half_t>) {
                    auto result = dequant_fp16(f);
                    tCrB_dequant(0, de_m, ik / 2) = result[0];
                    tCrB_dequant(2, de_m, ik / 2) = result[1];
                    tCrB_dequant(1, de_m, ik / 2) = result[2];
                    tCrB_dequant(3, de_m, ik / 2) = result[3];
                }
            }
            if (ik_next % 2 == 0) {
                cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next / 2, ismem_read), // tBsB: (CPY, CPY_M, CPY_K, kStage)
                        tCrB_view(_, _, ik_next / 2));                              // tCrB_view: (CPY, CPY_M, CPY_K)
            }       

            if (ik == 0) {
                for (; itile_to_read < ntile;) {
                    sparse_info_offset = sparse_kv_info_offset_base + itile_to_read * kTileK / head_dim;;
                    uint32_t sparse_info_offset_mod_8 = sparse_info_offset % 8;
                    if (sparse_info_offset / 8 != sparse_info_offset_loaded) {
                        sparse_info_offset_loaded = sparse_info_offset / 8;
                        uint8_t sparse_info_current_value = sparse_info_cur[sparse_info_offset_loaded];
                        for (int mod_id = sparse_info_offset_mod_8; mod_id < 8; ++mod_id) {
                            execute_bools[mod_id] = (sparse_info_current_value >> mod_id) & 1;
                        }
                    }
                    bool execute_this_tile = is_for_cached ? !execute_bools[sparse_info_offset_mod_8] : execute_bools[sparse_info_offset_mod_8];
                    // bool execute_this_tile = true;
                    if (execute_this_tile) {
                        break;
                    } else {
                        // itile_to_read += tile_per_head - (itile_to_read % tile_per_head);
                        itile_to_read += tile_per_head;
                    }
                }
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                            tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                            tBsB_copy(_, _, _, ismem_write));
                    cur_cached_stages++;
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB_dequant(_, _, 0), tCrD);
        }  // for ik
    }

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s);  // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);
            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

    #pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}


template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
            int kStage_ = 4, int kSmemLayoutCBatch_ = 2>
struct GemmConfig {
    using T = T_;

    // tile configuration
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kTileK_B_I8 = kTileK_ / 2; // 16 int8
    static constexpr int kStage = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    // using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_op = std::conditional_t<
        std::is_same_v<T, cute::half_t>,
        SM80_16x8x16_F32F16F16F32_TN,  // Type if T is cute::half_t
        SM80_16x8x16_F32BF16BF16F32_TN   // Type if T is not cute::half_t (e.g., bfloat16)
    >;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = typename mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    static constexpr int kThread = size(MMA{});
    // static constexpr int kThread = 256; // 8 warps

    static constexpr int kShmLoadSwizzleM = 3;
    static constexpr int kShmLoadSwizzleS = 3;
    static constexpr int kShmLoadSwizzleB = 3;

    // smem A layout
    using SmemALayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemALayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));

    // smem B layout
    using SmemBLayoutAtom = decltype(make_layout(
        make_shape(Int<kTileN>{}, Int<kTileK_B_I8>{}),
        LayoutRight{}));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemBLayoutAtom{},
        make_shape(Int<kTileN>{}, Int<kTileK_B_I8>{}, Int<kStage>{})));
        

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                                                make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
    using G2SCopyB = 
    decltype(make_tiled_copy(g2s_copy_atom{},
                                make_layout(make_shape(Int<64>{}, Int<2>{}), // Thr layout 32x4 k-major
                                            make_stride(Int<2>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8;

    // shared memory to register copy
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;
    

    // epilogue: register to global via shared memory
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                        make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than A's one pipe");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{}))));

    static constexpr int kThreadNum = size(MMA{});
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};



template <typename T>
cudaError_t SparseQKVGEMMReductionQuantRun(T* x, uint8_t* w, T* c, T* bias, 
                             uint32_t M, uint32_t N, uint32_t K, uint32_t batch_size,
                             uint32_t num_qo_heads, uint32_t head_dim,
                             uint8_t* sparse_info, uint32_t* sparse_info_indptr, 
                             uint32_t num_text_tokens,  bool is_for_cached, cudaStream_t stream) {
    
    GemmConfig<T, 128, 128, 32, 4> gemm_config;
    print("gemm_config: ");
    dim3 block = gemm_config.kThreadNum;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
                (M + gemm_config.kTileM - 1) / gemm_config.kTileM,
                batch_size);
    int shm_size = gemm_config.kShmSize;

    auto kernel = sparseqkv_gemm_reduction_quant_kernel<T, decltype(gemm_config)>;
    SPARSEQKV_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    // void* args[] = {x, w, c, M, N, K, sparse_info, sparse_info_indptr, is_single_stream_dit};
    // print("grid: "); print(grid.x); print(" "); print(grid.y); print(" "); print(grid.z); print("\n");
    void* args[] = {
        &x,                     // Address of the pointer x
        &w,                     // Address of the pointer w
        &bias,                  // Address of the pointer bias
        &c,                     // Address of the pointer c
        &M,                     // Address of the value M
        &N,                     // Address of the value N
        &K,                     // Address of the value K
        &num_qo_heads,         // Address of the value num_qo_heads
        &head_dim,            // Address of the value head_dim
        &sparse_info,           // Address of the pointer sparse_info
        &sparse_info_indptr,    // Address of the pointer sparse_info_indptr
        &num_text_tokens,   // Address of the value is_single_stream_dit
        &is_for_cached, // Address of the value is_for_cached
    };
    SPARSEQKV_CUDA_CALL(
        cudaLaunchKernel((void*)kernel, grid, block, args, shm_size, stream));
    
    return cudaSuccess;
}

}

}  // namespace sparseqkv

#endif  // SPARSEQKV_GEMM_CUH_