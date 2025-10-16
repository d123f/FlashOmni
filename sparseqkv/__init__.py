"""
Copyright (c) 2023 by FlashInfer team.

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

from ._build_meta import __version__ as __version__
from .mySparseFA import *

# from .attention_processor import _compute_sparse_info_indptr, _compute_sparse_kv_info_indptr 

from .gemm import (
    sparseqkv_gemm as sparseqkv_gemm,
    sparseqkv_gemm_reduction as sparseqkv_gemm_reduction,
    # sparseqkv_gemm_split as sparseqkv_gemm_split,
    # sparseqkv_gemm_reduction_quant as sparseqkv_gemm_reduction_quant,
    sparseqkv_gemm_int8 as sparseqkv_gemm_int8,
    # sparseqkv_gemm_to_q_int8 as sparseqkv_gemm_to_q_int8,
    )