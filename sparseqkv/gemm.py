"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .jit import (
    SPARSEQKV_CSRC_DIR,
    has_prebuilt_ops,
    load_cuda_ops
)

from .utils import (
    _get_cache_buf,
    register_custom_op,
    register_fake_op,
)

_gemm_module = None


def get_gemm_module():
    global _gemm_module
    if _gemm_module is None:
        if has_prebuilt_ops:
            _kernels = torch.ops.sparseqkv_kernels

            module = _kernels
        else:
            print(SPARSEQKV_CSRC_DIR)
            module = load_cuda_ops(
                "gemm",
                [
                    SPARSEQKV_CSRC_DIR / "gemm.cu",
                    SPARSEQKV_CSRC_DIR / "gemm_reduction.cu",
                    # SPARSEQKV_CSRC_DIR / "gemm_reduction_quant.cu",
                    # SPARSEQKV_CSRC_DIR / "gemm_split.cu",
                    SPARSEQKV_CSRC_DIR / "gemm_int8.cu",
                    SPARSEQKV_CSRC_DIR / "sparseqkv_gemm_ops.cu",
                ],
                extra_ldflags=["-lcublas", "-lcublasLt"],
            )

        # ====== FP16 & BF16 ======
        # torch library for sparseqkv_gemm
        @register_custom_op(
            "sparseqkv::sparseqkv_gemm", mutates_args=("bias")
        )
        def sparseqkv_gemm(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            sparse_q_size: int = 128,
            is_full: bool = False,
        ) -> None:
            # print(is_full)
            module.sparseqkv_gemm.default(
                A,
                B,
                out,
                bias,
                sparse_q_size,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                num_text_tokens,
                is_full,
            )

        @register_fake_op("sparseqkv::sparseqkv_gemm")
        def _fake_sparseqkv_gemm(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            sparse_q_size: int = 128,
            is_full: bool = False,
        ) -> None:
            pass

        # torch library for sparseqkv_gemm
        @register_custom_op(
            "sparseqkv::sparseqkv_gemm_reduction", mutates_args=("bias")
        )
        def sparseqkv_gemm_reduction(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            is_for_cache: bool = False,
            sparse_q_size: int = 128,
        ) -> None:
            module.sparseqkv_gemm_reduction.default(
                A,
                B,
                out,
                bias,
                sparse_q_size,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                num_text_tokens,
                is_for_cache
            )

        @register_fake_op("sparseqkv::sparseqkv_gemm_reduction")
        def _fake_sparseqkv_gemm_reduction(
            A: torch.Tensor,
            B: torch.Tensor,
            out: torch.Tensor,
            num_qo_heads: int,
            sparse_info: torch.Tensor,
            sparse_info_indptr: torch.Tensor,
            num_text_tokens: int,
            bias: torch.Tensor = None,
            is_for_cache: bool = False,
            sparse_q_size: int = 128,
        ) -> None:
            pass

        # # torch library for sparseqkv_gemm
        # @register_custom_op(
        #     "sparseqkv::sparseqkv_gemm_reduction_quant", mutates_args=("bias")
        # )
        # def sparseqkv_gemm_reduction_quant(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     out: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        #     is_for_cache: bool = False,
        # ) -> None:
        #     module.sparseqkv_gemm_reduction_quant.default(
        #         A,
        #         B,
        #         out,
        #         bias,
        #         num_qo_heads,
        #         sparse_info,
        #         sparse_info_indptr,
        #         num_text_tokens,
        #         is_for_cache,
        #     )

        # @register_fake_op("sparseqkv::sparseqkv_gemm_reduction_quant")
        # def _fake_sparseqkv_gemm_reduction_quant(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     out: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        #     is_for_cache: bool = False,
        # ) -> None:
        #     pass

        # # torch library for sparseqkv_gemm
        # @register_custom_op(
        #     "sparseqkv::sparseqkv_gemm_split", mutates_args=("bias")
        # )
        # def sparseqkv_gemm_split(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     out: torch.Tensor,
        #     cached_hidden: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        # ) -> None:
        #     module.sparseqkv_gemm_split.default(
        #         A,
        #         B,
        #         out,
        #         cached_hidden,
        #         bias,
        #         num_qo_heads,
        #         sparse_info,
        #         sparse_info_indptr,
        #         num_text_tokens,
        #     )

        # @register_fake_op("sparseqkv::sparseqkv_gemm_split")
        # def _fake_sparseqkv_gemm_split(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     out: torch.Tensor,
        #     cached_hidden: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        # ) -> None:
        #     pass

        # ===== W8A8O F16 ======
        # torch library for sparseqkv_gemm
        @register_custom_op(
            "sparseqkv::sparseqkv_gemm_int8", mutates_args=("bias")
        )
        def sparseqkv_gemm_int8(
            A: torch.Tensor,
            B: torch.Tensor,
            scale_input: torch.Tensor,
            scale_weight: torch.Tensor,
            input_sum: torch.Tensor,
            zp_weight: torch.Tensor,
            out: torch.Tensor,
            bias: torch.Tensor = None,
            num_qo_heads: int = None,
            sparse_info: torch.Tensor = None,
            sparse_info_indptr: torch.Tensor = None,
            num_text_tokens: int = None,
        ) -> None:
            module.sparseqkv_gemm_int8.default(
                A,
                B,
                out,
                bias,
                scale_input,
                scale_weight,
                input_sum,
                zp_weight,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                num_text_tokens,
            )

        @register_fake_op("sparseqkv::sparseqkv_gemm_int8")
        def _fake_sparseqkv_gemm_int8(
            A: torch.Tensor,
            B: torch.Tensor,
            scale_input: torch.Tensor,
            scale_weight: torch.Tensor,
            input_sum: torch.Tensor,
            zp_weight: torch.Tensor,
            out: torch.Tensor,
            bias: torch.Tensor = None,
            num_qo_heads: int = None,
            sparse_info: torch.Tensor = None,
            sparse_info_indptr: torch.Tensor = None,
            num_text_tokens: int = None,
        ) -> None:
            pass

        # # torch library for sparseqkv_gemm
        # @register_custom_op(
        #     "sparseqkv::sparseqkv_gemm_to_q_int8", mutates_args=("bias")
        # )
        # def sparseqkv_gemm_to_q_int8(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     scale_input: torch.Tensor,
        #     scale_weight: torch.Tensor,
        #     input_sum: torch.Tensor,
        #     zp_weight: torch.Tensor,
        #     out: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        # ) -> None:
        #     module.sparseqkv_gemm_int8.default(
        #         A,
        #         B,
        #         out,
        #         bias,
        #         scale_input,
        #         scale_weight,
        #         input_sum,
        #         zp_weight,
        #         num_qo_heads,
        #         sparse_info,
        #         sparse_info_indptr,
        #         num_text_tokens,
        #     )

        # @register_fake_op("sparseqkv::sparseqkv_gemm_to_q_int8")
        # def _fake_sparseqkv_gemm_to_q_int8(
        #     A: torch.Tensor,
        #     B: torch.Tensor,
        #     scale_input: torch.Tensor,
        #     scale_weight: torch.Tensor,
        #     input_sum: torch.Tensor,
        #     zp_weight: torch.Tensor,
        #     out: torch.Tensor,
        #     num_qo_heads: int,
        #     sparse_info: torch.Tensor,
        #     sparse_info_indptr: torch.Tensor,
        #     num_text_tokens: int,
        #     bias: torch.Tensor = None,
        # ) -> None:
        #     pass



        # Register the module
        _gemm_module = SimpleNamespace(
            sparseqkv_gemm = sparseqkv_gemm,
            sparseqkv_gemm_reduction = sparseqkv_gemm_reduction,
            # sparseqkv_gemm_reduction_quant = sparseqkv_gemm_reduction_quant,
            # sparseqkv_gemm_split = sparseqkv_gemm_split,
            # sparseqkv_gemm_to_q_int8 = sparseqkv_gemm_to_q_int8,
            sparseqkv_gemm_int8 = sparseqkv_gemm_int8,
        )

    return _gemm_module


# ====== FP16 & BF16 ======
def sparseqkv_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    num_qo_heads: int,
    sparse_info: torch.Tensor = None,
    sparse_info_indptr: torch.Tensor = None,
    num_text_tokens: int = 512,
    bias: torch.Tensor = None,
    out: torch.Tensor = None,
    sparse_q_size: int = 128,
    is_full: bool = False,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[0]),
            device=A.device,
            dtype=A.dtype,
        )
    get_gemm_module().sparseqkv_gemm(
        A,
        B,
        out,
        num_qo_heads,
        sparse_info,
        sparse_info_indptr,
        num_text_tokens,
        bias,
        sparse_q_size=sparse_q_size,
        is_full=is_full,
    )
    return out

def sparseqkv_gemm_reduction(
    A: torch.Tensor,
    B: torch.Tensor,
    num_qo_heads: int,
    sparse_info: torch.Tensor,
    sparse_info_indptr: torch.Tensor,
    num_text_tokens: int,
    bias: torch.Tensor = None,
    is_for_cache: bool = False,
    sparse_q_size: int = 128,
) -> torch.Tensor:
    out = torch.empty(
        (A.shape[0], A.shape[1], B.shape[0]),
        device=A.device,
        dtype=A.dtype,
    )
    get_gemm_module().sparseqkv_gemm_reduction(
        A,
        B,
        out,
        num_qo_heads,
        sparse_info,
        sparse_info_indptr,
        num_text_tokens,
        bias=bias,
        is_for_cache=is_for_cache,
        sparse_q_size=sparse_q_size,
    )
    return out

# def sparseqkv_gemm_reduction_quant(
#     A: torch.Tensor,
#     B: torch.Tensor,
#     num_qo_heads: int,
#     sparse_info: torch.Tensor,
#     sparse_info_indptr: torch.Tensor,
#     num_text_tokens: int,
#     bias: torch.Tensor = None,
#     is_for_cache: bool = False,
# ) -> torch.Tensor:
#     out = torch.zeros(
#         (A.shape[0], A.shape[1], B.shape[0]),
#         device=A.device,
#         dtype=A.dtype,
#     )
#     get_gemm_module().sparseqkv_gemm_reduction_quant(
#         A,
#         B,
#         out,
#         num_qo_heads,
#         sparse_info,
#         sparse_info_indptr,
#         num_text_tokens,
#         bias=bias,
#         is_for_cache=is_for_cache
#     )
#     return out

# def sparseqkv_gemm_split(
#     A: torch.Tensor,
#     B: torch.Tensor,
#     num_qo_heads: int,
#     sparse_info: torch.Tensor,
#     sparse_info_indptr: torch.Tensor,
#     num_text_tokens: int,
#     bias: torch.Tensor = None,
#     out: torch.Tensor = None,
#     cached_hidden: torch.Tensor = None,
# ) -> torch.Tensor:
#     if cached_hidden is None:
#         cached_hidden = torch.zeros(
#             (A.shape[0], A.shape[1], B.shape[0]),
#             device=torch.device('cuda'),
#             dtype=A.dtype,
#         )
#         print("cached_hidden", cached_hidden)
#     if out is None:
#         out = torch.zeros(
#             (A.shape[0], A.shape[1], B.shape[0]),
#             device=torch.device('cuda'),
#             dtype=A.dtype,
#         )
#         print("out", out)
#     get_gemm_module().sparseqkv_gemm_split(
#         A,
#         B,
#         out,
#         cached_hidden,
#         num_qo_heads,
#         sparse_info,
#         sparse_info_indptr,
#         num_text_tokens,
#         bias=bias,
#     )
#     print("cached_hidden", cached_hidden)
#     print("out", out)
#     return out, cached_hidden


# ====== W8A8O F16 ======
def sparseqkv_gemm_int8(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_input: torch.Tensor,
    scale_weight: torch.Tensor,
    input_sum: torch.Tensor,
    zp_weight: torch.Tensor,
    bias: torch.Tensor = None,
    num_qo_heads: int = None,
    sparse_info: torch.Tensor = None,
    sparse_info_indptr: torch.Tensor = None,
    num_text_tokens: int = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    if out is None:
        out = torch.zeros(
            (A.shape[0], A.shape[1], B.shape[0]),
            device=A.device,
            dtype=torch.float16,
        )
    get_gemm_module().sparseqkv_gemm_int8(
        A,
        B,
        scale_input,
        scale_weight,
        input_sum,
        zp_weight,
        out,
        bias,
        num_qo_heads,
        sparse_info,
        sparse_info_indptr,
        num_text_tokens,
    )
    return out

# def sparseqkv_gemm_to_q_int8(
#     A: torch.Tensor,
#     B: torch.Tensor,
#     scale_input: torch.Tensor,
#     scale_weight: torch.Tensor,
#     input_sum: torch.Tensor,
#     zp_weight: torch.Tensor,
#     num_qo_heads: int,
#     sparse_info: torch.Tensor,
#     sparse_info_indptr: torch.Tensor,
#     num_text_tokens: int,
#     bias: torch.Tensor = None,
#     out: torch.Tensor = None,
# ) -> torch.Tensor:
#     if out is None:
#         out = torch.empty(
#             (A.shape[0], A.shape[1], B.shape[0]),
#             device=A.device,
#             dtype=torch.float16,
#         )
#     get_gemm_module().sparseqkv_gemm_to_q_int8(
#         A,
#         B,
#         scale_input,
#         scale_weight,
#         input_sum,
#         zp_weight,
#         out,
#         num_qo_heads,
#         sparse_info,
#         sparse_info_indptr,
#         num_text_tokens,
#         bias,
#     )
#     return out
