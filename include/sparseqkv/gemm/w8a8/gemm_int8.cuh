#ifndef SPARSEQKV_GEMM_INT8_CUH_
#define SPARSEQKV_GEMM_INT8_CUH_

#include "../../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <cstdint>

#include "../../cp_async.cuh"
#include "../../mma.cuh"
#include "../../permuted_smem.cuh"
#include "gemm_utils.cuh"
#include "../../cutlass_utils.cuh"

#define PACK_SIZE_C16 8 // how many elements a load/store 128b instruction can manage for C

namespace sparseqkv {

namespace gemm {

template <uint32_t CTA_M, uint32_t CTA_N, uint32_t CTA_K, uint32_t WARP_M, uint32_t WARP_N,
       OutputDtype output_dtype=OutputDtype::kFloat16, uint32_t K_STAGE=2, bool has_bias=false, 
       bool weight_asym=false, ScaleMulMode scale_mul_mode=ScaleMulMode::kMode1, 
       GemmI8SparseMode gemm_i8_sparse_mode=GemmI8SparseMode::SMode1>
__global__ void sparseqkv_gemm_int8_kernel(const int8_t *__restrict__ A, const int8_t *__restrict__ B,
                              half *__restrict__ C16, const half *__restrict__ Bias, 
                              const half *__restrict__ scale_A, const half *__restrict__ scale_B,
                              const half *__restrict__ sum_A, const int16_t *__restrict__ zp_B,
                              const int M, const int N, const int K, uint32_t num_qo_heads, uint32_t head_dim,
                              uint8_t* sparse_info, uint32_t* sparse_info_indptr, uint32_t num_text_tokens)
{
  static_assert(K_STAGE > 1);
  constexpr uint32_t num_warps_m = CTA_M / WARP_M;
  constexpr uint32_t num_warps_n = CTA_N / WARP_N;
  constexpr uint32_t num_warps = num_warps_m * num_warps_n;
  constexpr uint32_t num_tiles_m = WARP_M / MMA_M_i8;
  constexpr uint32_t num_tiles_n = WARP_N / MMA_N_i8;
  constexpr uint32_t num_tiles_k = CTA_K / MMA_K_i8;

  static_assert(num_tiles_k % 2 == 0);

  constexpr uint32_t AB_SMEM_STRIDE = CTA_K;
  constexpr uint32_t C_SMEM_STRIDE = CTA_N;

  uint32_t blockIdx_m = blockIdx.y;
  uint32_t blockIdx_n = blockIdx.z;
  uint32_t batchIdx = blockIdx.x;

  if (blockIdx_m >= M / CTA_M || blockIdx_n >= N / CTA_N)
  {
    return;
  }

  if constexpr (gemm_i8_sparse_mode == GemmI8SparseMode::SMode2) {
    uint8_t* sparse_info_cur = sparse_info + sparse_info_indptr[batchIdx];
    // B, N, H
    const uint32_t sparse_info_offset = (num_text_tokens + blockIdx_m * CTA_M) / 128 * num_qo_heads
                            + blockIdx_n * CTA_N / head_dim;

    if (!((sparse_info_cur[sparse_info_offset / 8] >> (sparse_info_offset % 8)) & 1)) {
      return;
    }
  }

  extern __shared__ int8_t smem[][AB_SMEM_STRIDE];

  const uint32_t warp_id = get_warp_id();
  const uint32_t lane_id = get_lane_id();

  // RC holds the fragment of C
  int32_t RC[num_tiles_m][num_tiles_n][8];

  // initialize RC
#pragma unroll
  for (uint32_t i = 0; i < num_tiles_m; ++i)
  {
#pragma unroll
    for (uint32_t j = 0; j < num_tiles_n; ++j)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; ++k)
      {
        RC[i][j][k] = 0;
      }
    }
  }

  constexpr uint32_t B_smem_idx_off = CTA_M;
  constexpr uint32_t smem_stage_off = CTA_M + CTA_N;

  constexpr SwizzleMode swizzle_mode_AB = (AB_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_AB> current_smem_A(smem);
  smem_t<swizzle_mode_AB> current_smem_B(smem + B_smem_idx_off);

  constexpr SwizzleMode swizzle_mode_C16 = (C_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_C16> smem_C16(smem);

  // A_warp_base_ptr and B_warp_base_ptr are used to load data from global memory to shared memory
  // each warp loads a few rows for A
  const int8_t *A_warp_base_ptr = A + batchIdx * M * K + blockIdx_m * CTA_M * K + CTA_M / num_warps * warp_id * K;
  const int8_t *B_warp_base_ptr = B + blockIdx_n * CTA_N * K + CTA_N / num_warps * warp_id * K;
  half *C16_warp_base_ptr = C16 + batchIdx * M * N + blockIdx_m * CTA_M * N + CTA_M / num_warps * warp_id * N + blockIdx_n * CTA_N;
  
  constexpr uint32_t global_to_shared_line_lanes = (AB_SMEM_STRIDE == 64) ? 4 : 8; // when loading from global to shared memory, how many lanes are used to load a line
  constexpr uint32_t global_to_shared_copy_lines_per_warp = (AB_SMEM_STRIDE == 64) ? 8 : 4; // how many lines are copied per warp per iteration

  constexpr uint32_t global_to_shared_line_lanes_C16 = (C_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_C16 = (C_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t A_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE_i8);
  constexpr uint32_t B_smem_iters_row = AB_SMEM_STRIDE / (global_to_shared_line_lanes * PACK_SIZE_i8);
  constexpr uint32_t A_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp);
  constexpr uint32_t B_smem_iters_col = CTA_N / (num_warps * global_to_shared_copy_lines_per_warp);

  constexpr uint32_t C16_smem_iters_row = C_SMEM_STRIDE / (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
  constexpr uint32_t C16_smem_iters_col = CTA_M / (num_warps * global_to_shared_copy_lines_per_warp_C16);


  // store idx is used to store data to shared memory
  uint32_t smem_store_idx = K_STAGE - 1, smem_store_off = 0;
  // load idx is used to load data from shared memory to registers
  uint32_t smem_load_idx = 0, smem_load_off = 0;
  uint32_t cur_cached_stages = 0, offset_K = 0;

  // prefetch K_STAGE stages of data
#pragma unroll
  for (uint32_t stage = 0; stage < K_STAGE - 1; stage++)
  {
    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    current_smem_A.set_base(smem + smem_store_off);
    current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
      A_warp_base_ptr + offset_K, K, current_smem_A);
    load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
      B_warp_base_ptr + offset_K, K, current_smem_B);

    cp_async::commit_group();
    offset_K += CTA_K;
    cur_cached_stages++;
  }

  // ensure stage 0 is ready
  cp_async::wait_group<K_STAGE - 2>();
  __syncthreads();

  // store_idx is used to store data to register
  uint32_t reg_store_idx = 0;
  // load_idx is used to load data from register to tensor core
  uint32_t reg_load_idx = 1;
  
  uint32_t RA[2][num_tiles_m][4];
  uint32_t RB[2][num_tiles_n][4];


  current_smem_A.set_base(smem + smem_load_off);
  current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

  share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
    current_smem_A, RA[reg_store_idx], 0);
  share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
    current_smem_B, RB[reg_store_idx], 0);

  while (cur_cached_stages != 0) {
#pragma unroll
    for (uint32_t k = 0; k < num_tiles_k; k += 1) {
      uint32_t ik_next = (k + 1) % num_tiles_k;
      if (k == num_tiles_k - 1) {
        cp_async::wait_group<K_STAGE - 2>();
        __syncthreads();
        cur_cached_stages--;

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;
      }

      reg_store_idx ^= 1; // reg_store_idx is 0 here
      reg_load_idx ^= 1; // reg_load_idx is 1 here

      current_smem_A.set_base(smem + smem_load_off);
      current_smem_B.set_base(smem + smem_load_off + B_smem_idx_off);

      share_to_reg_A<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
        current_smem_A, RA[reg_store_idx], 2 * ik_next);
      share_to_reg_B<num_warps_m, num_warps_n, num_tiles_m, num_tiles_n, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
        current_smem_B, RB[reg_store_idx], 2 * ik_next);


      if (k == 0) {
        if (offset_K < K) {
          smem_store_idx = (smem_store_idx + 1) % K_STAGE;
          smem_store_off = smem_store_idx * smem_stage_off;

          current_smem_A.set_base(smem + smem_store_off);
          current_smem_B.set_base(smem + smem_store_off + B_smem_idx_off);

          load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, A_smem_iters_row, A_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
            A_warp_base_ptr + offset_K, K, current_smem_A);
          load_AB_global_smem<global_to_shared_line_lanes, global_to_shared_copy_lines_per_warp, B_smem_iters_row, B_smem_iters_col, swizzle_mode_AB, AB_SMEM_STRIDE / PACK_SIZE_i8>(
            B_warp_base_ptr + offset_K, K, current_smem_B);
          offset_K += CTA_K;
          cur_cached_stages++;
        }
        cp_async::commit_group();
      }

      tensor_core_mma<num_tiles_m, num_tiles_n>(RC, RA[reg_load_idx], RB[reg_load_idx]);
    }
  }

  // fp16 output
  if constexpr (output_dtype == OutputDtype::kFloat16)
  {
    // not well optimized, but this part is not bottleneck
    const half *scale_A_warp_ptr = scale_A + batchIdx * M + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const half *scale_B_warp_ptr = scale_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const int16_t *zp_B_warp_ptr = zp_B + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;
    const half *sum_a_warp_ptr = sum_A + batchIdx * M + blockIdx_m * CTA_M + get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M;
    const half *bias_warp_ptr = Bias + blockIdx_n * CTA_N + get_warp_idx_n<num_warps_m, num_warps_n>() * WARP_N;

    float a_scale = 1.0f;
    float2 b_scale = {1.0f, 1.0f};
    float a_sum = 0.0f;
    short2 zp_b = {0, 0};
    float2 bias = {0.0f, 0.0f};
    float2 psums = {0.0f, 0.0f};
#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        uint32_t a_scale_load = 0;
#pragma unroll
        for (uint32_t k = 0; k < 8; k += 2) {
          a_scale = __half2float(*(scale_A_warp_ptr + i * MMA_M_i8 + lane_id / 4 + a_scale_load * 8));
          if (k % 4 == 0) {
            b_scale = __half22float2(*reinterpret_cast<const half2*>(scale_B_warp_ptr + j * MMA_N_i8 + 2 * (lane_id % 4) + 2 * k));
          }
          psums = make_float2(__int2float_rn(RC[i][j][k]), __int2float_rn(RC[i][j][k + 1]));
          if constexpr (scale_mul_mode == ScaleMulMode::kMode1)
          {
            psums.x = psums.x * a_scale * b_scale.x;
            psums.y = psums.y * a_scale * b_scale.y;
          }
          else if constexpr (scale_mul_mode == ScaleMulMode::kMode2)
          {
            psums.x *= a_scale * b_scale.x;
            psums.y *= a_scale * b_scale.y; 
          }
          if constexpr (weight_asym)
          {
            a_sum = __half2float(*(sum_a_warp_ptr + i * MMA_M_i8 + lane_id / 4 + a_scale_load * 8));
            if (k % 4 == 0) {
              zp_b = *reinterpret_cast<const short2*>(zp_B_warp_ptr + j * MMA_N_i8 + 2 * (lane_id % 4) + 2 * k);
            }
            psums.x = psums.x + a_sum * static_cast<float>(zp_b.x) * b_scale.x;
            psums.y = psums.y + a_sum * static_cast<float>(zp_b.y) * b_scale.y;
          }
          if constexpr (has_bias)
          {
            if (k % 4 == 0) {
              bias = __half22float2(*reinterpret_cast<const half2*>(bias_warp_ptr + j * MMA_N_i8 + 2 * (lane_id % 4) + 2 * k));
            }
            psums.x += bias.x;
            psums.y += bias.y;
          }
          ((half2*)RC[i][j])[k] = __float22half2_rn(psums);
          a_scale_load ^= 1;
        }
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < num_tiles_m; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < num_tiles_n; j++)
      {
        uint32_t offset_C1 = smem_C16.get_permuted_offset<C_SMEM_STRIDE / PACK_SIZE_C16>(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M_i8 + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N_i8 / PACK_SIZE_C16));
        
        ((int32_t*)(smem_C16.base + offset_C1))[lane_id % 4] = RC[i][j][0];
        ((int32_t*)(smem_C16.base + offset_C1 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][2];

        uint32_t offset_C2 = smem_C16.get_permuted_offset<C_SMEM_STRIDE / PACK_SIZE_C16>(
          get_warp_idx_m<num_warps_m, num_warps_n>() * WARP_M + i * MMA_M_i8 + lane_id / 4,
          get_warp_idx_n<num_warps_m, num_warps_n>() * (WARP_N / PACK_SIZE_C16) + j * (MMA_N_i8 / PACK_SIZE_C16) + 1);
        
        ((int32_t*)(smem_C16.base + offset_C2))[lane_id % 4] = RC[i][j][4];
        ((int32_t*)(smem_C16.base + offset_C2 + 8 * (C_SMEM_STRIDE / PACK_SIZE_C16)))[lane_id % 4] = RC[i][j][6];
      }
    }

    __syncthreads();

    half *C_lane_ptr = C16_warp_base_ptr + lane_id / global_to_shared_line_lanes_C16 * N + lane_id % global_to_shared_line_lanes_C16 * PACK_SIZE_C16;
    uint32_t offset_C = smem_C16.get_permuted_offset<C_SMEM_STRIDE / PACK_SIZE_C16>(warp_id * global_to_shared_copy_lines_per_warp_C16 * C16_smem_iters_col + lane_id / global_to_shared_line_lanes_C16, lane_id % global_to_shared_line_lanes_C16);

#pragma unroll
    for (uint32_t i = 0; i < C16_smem_iters_col; i++)
    {
#pragma unroll
      for (uint32_t j = 0; j < C16_smem_iters_row; j++)
      {
        smem_C16.store_128b(offset_C, C_lane_ptr);
        C_lane_ptr += (global_to_shared_line_lanes_C16 * PACK_SIZE_C16);
        offset_C = smem_C16.advance_offset_by_column<global_to_shared_line_lanes_C16>(offset_C);
      }

      offset_C = smem_C16.advance_offset_by_row<global_to_shared_copy_lines_per_warp_C16, C_SMEM_STRIDE / PACK_SIZE_C16>(offset_C - (C16_smem_iters_row * global_to_shared_line_lanes_C16));
      C_lane_ptr += ((global_to_shared_copy_lines_per_warp_C16 * N) - (C16_smem_iters_row * global_to_shared_line_lanes_C16 * PACK_SIZE_C16));
    }

  }

}


cudaError_t SparseQKVGEMMInt8Run(int8_t* input, int8_t* weight, half* output, half* bias, 
                              half* scale_input, half* scale_weight, half* sum_input, int16_t* zp_weight,
                             uint32_t M, uint32_t N, uint32_t K, uint32_t batch_size,
                             uint32_t num_qo_heads, uint32_t head_dim,
                             uint8_t* sparse_info, uint32_t* sparse_info_indptr, 
                             uint32_t num_text_tokens, cudaStream_t stream) {

    const int CTA_M = 128;
    const int CTA_N = 128;
    const int CTA_K = 64;
    constexpr int WARP_M = 128;
    constexpr int WARP_N = 32;
    constexpr int K_STAGE = 3;

    assert(M % CTA_M == 0);
    assert(N % CTA_N == 0);
    assert(K % CTA_K == 0);
                          
    size_t shm_size = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(half));
    
    dim3 grid(batch_size, M / CTA_M, N / CTA_N);
    dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));
    void* args[] = {
        &input,                     // Address of the pointer x
        &weight,                     // Address of the pointer w
        &output,                     // Address of the pointer c
        &bias,                  // Address of the pointer bias
        &scale_input,            // Address of the pointer scale_input
        &scale_weight,            // Address of the pointer scale_weight
        &sum_input,            // Address of the pointer sum_input
        &zp_weight,
        &M,                     // Address of the value M
        &N,                     // Address of the value N
        &K,                     // Address of the value K
        &num_qo_heads,         // Address of the value num_qo_heads
        &head_dim,            // Address of the value head_dim
        &sparse_info,           // Address of the pointer sparse_info
        &sparse_info_indptr,    // Address of the pointer sparse_info_indptr
        &num_text_tokens,   // Address of the value is_single_stream_dit
    };
    if (sparse_info == nullptr) {
      auto kernel = sparseqkv_gemm_int8_kernel<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, OutputDtype::kFloat16, K_STAGE, true, true, ScaleMulMode::kMode1, GemmI8SparseMode::SMode1>;

      SPARSEQKV_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      
      
      SPARSEQKV_CUDA_CALL(
          cudaLaunchKernel((void*)kernel, grid, block, args, shm_size, stream));
    } else {
      auto kernel = sparseqkv_gemm_int8_kernel<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, OutputDtype::kFloat16, K_STAGE, true, true, ScaleMulMode::kMode1, GemmI8SparseMode::SMode2>;

      SPARSEQKV_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      
      SPARSEQKV_CUDA_CALL(
          cudaLaunchKernel((void*)kernel, grid, block, args, shm_size, stream));
    }
    
    return cudaSuccess;
}


// cudaError_t SparseQKVGEMMInt8ToQRun(int8_t* input, int8_t* weight, half* output, half* bias, 
//                               half* scale_input, half* scale_weight, half* sum_input, int16_t* zp_weight,
//                              uint32_t M, uint32_t N, uint32_t K, uint32_t batch_size,
//                              uint32_t num_qo_heads, uint32_t head_dim,
//                              uint8_t* sparse_info, uint32_t* sparse_info_indptr, 
//                              uint32_t num_text_tokens, cudaStream_t stream) {

//     const int CTA_M = 128;
//     const int CTA_N = 128;
//     const int CTA_K = 64;
//     constexpr int WARP_M = 128;
//     constexpr int WARP_N = 32;
//     constexpr int K_STAGE = 3;

//     assert(M % CTA_M == 0);
//     assert(N % CTA_N == 0);
//     assert(K % CTA_K == 0);
                          
//     size_t shm_size = std::max((CTA_M * CTA_K + CTA_N * CTA_K) * sizeof(int8_t) * K_STAGE, CTA_M * CTA_N * sizeof(half));

//     auto kernel = sparseqkv_gemm_int8_kernel<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, OutputDtype::kFloat16, K_STAGE, true, true, ScaleMulMode::kMode1, GemmI8SparseMode::SMode2>;

//     SPARSEQKV_CUDA_CALL(
//         cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
//     // void* args[] = {x, w, c, M, N, K, sparse_info, sparse_info_indptr, is_single_stream_dit};
    
//     dim3 grid(batch_size, M / CTA_M, N / CTA_N);
//     dim3 block(32, (CTA_M / WARP_M) * (CTA_N / WARP_N));
//     void* args[] = {
//         &input,                     // Address of the pointer x
//         &weight,                     // Address of the pointer w
//         &output,                     // Address of the pointer c
//         &bias,                  // Address of the pointer bias
//         &scale_input,            // Address of the pointer scale_input
//         &scale_weight,            // Address of the pointer scale_weight
//         &sum_input,            // Address of the pointer sum_input
//         &zp_weight,
//         &M,                     // Address of the value M
//         &N,                     // Address of the value N
//         &K,                     // Address of the value K
//         &num_qo_heads,         // Address of the value num_qo_heads
//         &head_dim,            // Address of the value head_dim
//         &sparse_info,           // Address of the pointer sparse_info
//         &sparse_info_indptr,    // Address of the pointer sparse_info_indptr
//         &num_text_tokens,   // Address of the value is_single_stream_dit
//     };
//     SPARSEQKV_CUDA_CALL(
//         cudaLaunchKernel((void*)kernel, grid, block, args, shm_size, stream));
    
//     return cudaSuccess;
// }

}



}  // namespace sparseqkv


#endif  // SPARSEQKV_GEMM_CUH_