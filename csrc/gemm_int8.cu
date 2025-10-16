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
#include <cstddef>
#include <cstdint>
#include <sparseqkv/gemm/w8a8/gemm_int8.cuh>
// #include <sparseqkv/gemm/w8a8/gemm_int8_to_out.cuh>

#include "pytorch_extension_utils.h"

using namespace sparseqkv;
using namespace sparseqkv::gemm;

void SparseQKVGEMMInt8(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
              at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight,
              std::optional<int64_t> num_qo_heads, std::optional<at::Tensor> sparse_info, 
              std::optional<at::Tensor> sparse_info_indptr, std::optional<int64_t> num_text_tokens) {
  unsigned int batch_size = x_ptr.size(0);
  unsigned int M = x_ptr.size(1);
  unsigned int K = x_ptr.size(2);
  unsigned int N = w_ptr.size(0); // weight = (N, K)

  TORCH_CHECK(x_ptr.size(0) == y_ptr.size(0), "Batch sizes must match");
  TORCH_CHECK(x_ptr.size(1) == y_ptr.size(1), "Token sizes must match");
  TORCH_CHECK(x_ptr.size(2) == w_ptr.size(1) && w_ptr.size(0) == y_ptr.size(2),
              "Result tensor has incorrect shape");
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "only support M % 128 == 0, N % 128 == 0, K % 32 == 0");
  
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_INT8(x_ptr.scalar_type(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;

    half* ptr_to_bias = bias.has_value()? 
             reinterpret_cast<half*>(bias.value().data_ptr())  : nullptr;

    int64_t num_qo_heads_ = num_qo_heads.has_value()? num_qo_heads.value() : 1;

    uint8_t* sparse_info_ = sparse_info.has_value() ? 
             reinterpret_cast<uint8_t*>(sparse_info.value().data_ptr()) : nullptr;

    uint32_t* sparse_info_indptr_ = sparse_info_indptr.has_value() ?
            reinterpret_cast<uint32_t*>(sparse_info_indptr.value().data_ptr()) : nullptr;

    int64_t num_text_tokens_ = num_text_tokens.has_value() ? num_text_tokens.value() : 0;
    unsigned int head_dim = N / num_qo_heads_;

    auto status = SparseQKVGEMMInt8Run(
        reinterpret_cast<cutlass_t*>(x_ptr.data_ptr()), reinterpret_cast<cutlass_t*>(w_ptr.data_ptr()), 
        reinterpret_cast<half*>(y_ptr.data_ptr()),  ptr_to_bias, reinterpret_cast<half*>(scale_input.data_ptr()),
        reinterpret_cast<half*>(scale_weight.data_ptr()), reinterpret_cast<half*>(sum_input.data_ptr()),
        reinterpret_cast<int16_t*>(zp_weight.data_ptr()), M, N, K, batch_size, 
        num_qo_heads_, head_dim, sparse_info_, sparse_info_indptr_, num_text_tokens_, stream);
    TORCH_CHECK(status == cudaSuccess,
          "Failed to run SparseQKVGEMMInt8: ", cudaGetErrorString(status));
    return true;
  });
}


// void SparseQKVGEMMInt8ToQ(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
//               at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight,
//               int64_t num_qo_heads, at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens) {
//   unsigned int batch_size = x_ptr.size(0);
//   unsigned int M = x_ptr.size(1);
//   unsigned int K = x_ptr.size(2);
//   unsigned int N = w_ptr.size(0); // weight = (N, K)

//   unsigned int head_dim = N / static_cast<uint32_t>(num_qo_heads);
//   TORCH_CHECK(x_ptr.size(0) == y_ptr.size(0), "Batch sizes must match");
//   TORCH_CHECK(x_ptr.size(1) == y_ptr.size(1), "Token sizes must match");
//   TORCH_CHECK(x_ptr.size(2) == w_ptr.size(1) && w_ptr.size(0) == y_ptr.size(2),
//               "Result tensor has incorrect shape");
//   TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "only support M % 128 == 0, N % 128 == 0, K % 32 == 0");
  
//   auto stream = at::cuda::getCurrentCUDAStream();

//   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_INT8(x_ptr.scalar_type(), c_type, [&] {
//     using cutlass_t = cutlass_dtype_t<c_type>;
//     half* ptr_to_bias = bias.has_value()? 
//              reinterpret_cast<half*>(bias.value().data_ptr())  : nullptr;
//     auto status = SparseQKVGEMMInt8ToQRun(
//         reinterpret_cast<cutlass_t*>(x_ptr.data_ptr()), reinterpret_cast<cutlass_t*>(w_ptr.data_ptr()), 
//         reinterpret_cast<half*>(y_ptr.data_ptr()),  ptr_to_bias, reinterpret_cast<half*>(scale_input.data_ptr()),
//         reinterpret_cast<half*>(scale_weight.data_ptr()), reinterpret_cast<half*>(sum_input.data_ptr()),
//         reinterpret_cast<int16_t*>(zp_weight.data_ptr()),
//         M, N, K, batch_size, num_qo_heads, head_dim, reinterpret_cast<uint8_t*>(sparse_info.data_ptr()), 
//         reinterpret_cast<uint32_t*>(sparse_info_indptr.data_ptr()), num_text_tokens, stream);
//     TORCH_CHECK(status == cudaSuccess,
//           "Failed to run SparseQKVGEMM: ", cudaGetErrorString(status));
//     return true;
//   });
// }


// void SparseQKVGEMMInt8(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
//               at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight) {
//   unsigned int batch_size = x_ptr.size(0);
//   unsigned int M = x_ptr.size(1);
//   unsigned int K = x_ptr.size(2);
//   unsigned int N = w_ptr.size(0); // weight = (N, K)

//   TORCH_CHECK(x_ptr.size(0) == y_ptr.size(0), "Batch sizes must match");
//   TORCH_CHECK(x_ptr.size(1) == y_ptr.size(1), "Token sizes must match");
//   TORCH_CHECK(x_ptr.size(2) == w_ptr.size(1) && w_ptr.size(0) == y_ptr.size(2),
//               "Result tensor has incorrect shape");
//   TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "only support M % 128 == 0, N % 128 == 0, K % 32 == 0");
  
//   auto stream = at::cuda::getCurrentCUDAStream();

//   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_INT8(x_ptr.scalar_type(), c_type, [&] {
//     using cutlass_t = cutlass_dtype_t<c_type>;
//     half* ptr_to_bias = bias.has_value()? 
//              reinterpret_cast<half*>(bias.value().data_ptr())  : nullptr;
//     auto status = SparseQKVGEMMInt8Run(
//         reinterpret_cast<cutlass_t*>(x_ptr.data_ptr()), reinterpret_cast<cutlass_t*>(w_ptr.data_ptr()), 
//         reinterpret_cast<half*>(y_ptr.data_ptr()),  ptr_to_bias, reinterpret_cast<half*>(scale_input.data_ptr()),
//         reinterpret_cast<half*>(scale_weight.data_ptr()), reinterpret_cast<half*>(sum_input.data_ptr()),
//         reinterpret_cast<int16_t*>(zp_weight.data_ptr()), M, N, K, batch_size, stream);
//     TORCH_CHECK(status == cudaSuccess,
//           "Failed to run SparseQKVGEMM: ", cudaGetErrorString(status));
//     return true;
//   });

// }


// void SparseQKVGEMMInt8ToQ(at::Tensor x_ptr, at::Tensor w_ptr, at::Tensor y_ptr, std::optional<at::Tensor> bias, 
//               at::Tensor scale_input, at::Tensor scale_weight, at::Tensor sum_input, at::Tensor zp_weight,
//               int64_t num_qo_heads, at::Tensor sparse_info, at::Tensor sparse_info_indptr, int64_t num_text_tokens) {
//   unsigned int batch_size = x_ptr.size(0);
//   unsigned int M = x_ptr.size(1);
//   unsigned int K = x_ptr.size(2);
//   unsigned int N = w_ptr.size(0); // weight = (N, K)

//   unsigned int head_dim = N / static_cast<uint32_t>(num_qo_heads);
//   TORCH_CHECK(x_ptr.size(0) == y_ptr.size(0), "Batch sizes must match");
//   TORCH_CHECK(x_ptr.size(1) == y_ptr.size(1), "Token sizes must match");
//   TORCH_CHECK(x_ptr.size(2) == w_ptr.size(1) && w_ptr.size(0) == y_ptr.size(2),
//               "Result tensor has incorrect shape");
//   TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0, "only support M % 128 == 0, N % 128 == 0, K % 32 == 0");
  
  
//   auto stream = at::cuda::getCurrentCUDAStream();

//   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_INT8(x_ptr.scalar_type(), c_type, [&] {
//     using cutlass_t = cutlass_dtype_t<c_type>;
//     half* ptr_to_bias = bias.has_value()? 
//              reinterpret_cast<half*>(bias.value().data_ptr())  : nullptr;
//     auto status = SparseQKVGEMMInt8ToQRun(
//         reinterpret_cast<cutlass_t*>(x_ptr.data_ptr()), reinterpret_cast<cutlass_t*>(w_ptr.data_ptr()), 
//         reinterpret_cast<half*>(y_ptr.data_ptr()),  ptr_to_bias, reinterpret_cast<half*>(scale_input.data_ptr()),
//         reinterpret_cast<half*>(scale_weight.data_ptr()), reinterpret_cast<half*>(sum_input.data_ptr()),
//         reinterpret_cast<int16_t*>(zp_weight.data_ptr()),
//         M, N, K, batch_size, num_qo_heads, head_dim, reinterpret_cast<uint8_t*>(sparse_info.data_ptr()), 
//         reinterpret_cast<uint32_t*>(sparse_info_indptr.data_ptr()), num_text_tokens, stream);
//     TORCH_CHECK(status == cudaSuccess,
//           "Failed to run SparseQKVGEMM: ", cudaGetErrorString(status));
//     return true;
//   });
// }