/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "pytorch_extension_utils.h"

void SparseQKVGEMM(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  std::optional<at::Tensor> sparse_info, std::optional<at::Tensor> sparse_info_indptr, int64_t num_text_tokens, bool is_full);

void SparseQKVGEMMReduction(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t sparse_q_size, int64_t num_qo_heads,
  at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens, bool is_for_cached);

// void SparseQKVGEMMReductionQuant(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, int64_t num_qo_heads,
//   at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens, bool is_for_cached);

// void SparseQKVGEMMSplit(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, at::Tensor cached_hidden_ptr,
//   std::optional<at::Tensor> bias, int64_t num_qo_heads, at::Tensor sparse_info, 
//   at::Tensor sparse_info_indptr, int64_t num_text_tokens);
  
// void SparseQKVGEMMInt8ToQ(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
//               at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight,
//               int64_t num_qo_heads, at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens);

void SparseQKVGEMMInt8(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
              at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight,
              std::optional<int64_t> num_qo_heads, std::optional<at::Tensor> sparse_info, 
              std::optional<at::Tensor> sparse_info_indptr, std::optional<int64_t> num_text_tokens);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("sparseqkv_gemm", SparseQKVGEMM);
  m.def("sparseqkv_gemm_reduction", SparseQKVGEMMReduction);
  // m.def("sparseqkv_gemm_reduction_quant", SparseQKVGEMMReductionQuant);
  // m.def("sparseqkv_gemm_split", SparseQKVGEMMSplit);
  // m.def("sparseqkv_gemm_to_q_int8", SparseQKVGEMMInt8ToQ);
  m.def("sparseqkv_gemm_int8", SparseQKVGEMMInt8);
}
