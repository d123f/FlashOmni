/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SPARSEQKV_ATTENTION_VARIANTS_CUH_
#define SPARSEQKV_ATTENTION_VARIANTS_CUH_
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "../math.cuh"
#include "../utils.cuh"
#include "variant_helper.cuh"

namespace sparseqkv {

DEFINE_HAS_MEMBER(maybe_mask_indptr)

template <bool use_custom_mask, bool use_logits_soft_cap, bool use_alibi>
struct DefaultAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  // uint8_t* sparse_kv_info;
  uint32_t qo_len, kv_len;
  float sm_scale_log2;
  float soft_cap_pre_tanh_scale;

  // Create closure
  template <typename Params>
  __device__ __host__ DefaultAttention(const Params& params, uint32_t batch_idx,
                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_logits_soft_cap) {
      soft_cap_pre_tanh_scale = params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
      sm_scale_log2 = math::log2e * params.logits_soft_cap;
    } else {
      if constexpr (use_alibi) {
        sm_scale_log2 = math::log2e;
      } else {
        sm_scale_log2 = params.sm_scale * math::log2e;
      }
    }
    if constexpr (use_custom_mask) {
      if constexpr (has_maybe_mask_indptr_v<Params>) {
        custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
      } else {
        custom_mask_ptr = params.maybe_custom_mask;
      }
    }
    // sparse_kv_info = params.sparse_kv_info + params.sparse_kv_info_indptr[batch_idx];
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if constexpr (use_alibi) {
      logits = logits * params.sm_scale +
               params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      logits = float(math::tanh(logits * soft_cap_pre_tanh_scale));
    }
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if constexpr (use_custom_mask) {
      if (qo_idx >= qo_len || kv_idx >= kv_len) {
        mask = false;
      } else {
        const uint32_t offset = qo_idx * kv_len + kv_idx;
        mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
      }
    }
    return mask;
  })

  // REGISTER_Sparse_KV_Info(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
  //   bool execute = true;
  //   // const uint32_t offset = (qo_head_idx + qo_idx / params.sparse_block_size_for_q * params.num_qo_heads);  // (N_q * H* sparse_num + Head *sparse_num + N_k )
  //   // execute &= ((sparse_info[offset / 8] >> (offset % 8)) & 1);
  //   return execute;
  // })

};

};  // namespace sparseqkv

#endif  // SPARSEQKV_ATTENTION_VARIANTS_CUH_
