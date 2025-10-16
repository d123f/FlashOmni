import torch
from typing import Optional
from diffusers.models.attention_processor import Attention
from typing import Any, Dict, Optional
from sparseqkv.gemm import sparseqkv_gemm, sparseqkv_gemm_reduction
from flashinfer import segment_packbits
import torch.nn.functional as F
import triton
import triton.language as tl
from ..taylorseer_utils import derivative_approximation, taylor_formula, saving_sparse_info


@triton.jit
def triton_fill_sparseqkv_kernel(sparse_kv_info, sparse_info, 
                                 num_kv_to_select, sorted_kv_indices, 
                                 num_to_select, sorted_indices, 
                                 NK: tl.constexpr, T2T: tl.constexpr = 0):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    if q == 0:
        # process_sparse_info
        cur_num_to_select = tl.load(num_to_select + b * H + h)
        cur_sorted_idx_ptr = sorted_indices + b * Q * H
        cur_final_map_ptr = sparse_info + b * H * Q + h * Q + T2T
        for i in range(cur_num_to_select):
            cur_idx = tl.load(cur_sorted_idx_ptr + i)
            tl.store(cur_final_map_ptr + cur_idx, 0)
    # process_sparse_kv_info
    # elif q >= T2T:
    cur_num_kv_to_select = tl.load(num_kv_to_select + b * H * Q + h * Q + q)
    cur_sorted_kv_idx_ptr = sorted_kv_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_sparse_kv_info_ptr = sparse_kv_info + b * H * Q * NK + h * Q* NK + q * NK 
    for i in range(cur_num_kv_to_select):
        cur_idx = tl.load(cur_sorted_kv_idx_ptr + i)
        tl.store(cur_sparse_kv_info_ptr + cur_idx, 0)

def fill_sparseqkv_triton(sparse_kv_info, num_kv_to_select, sorted_kv_indices,
                          sparse_info, num_q_to_select, sorted_q_indices, T2T=0):
    sparse_kv_info = sparse_kv_info.contiguous()
    num_kv_to_select = num_kv_to_select.contiguous()
    sorted_kv_indices = sorted_kv_indices.contiguous()
    B, H, NBlock_Q, NBlock_KV = sparse_kv_info.shape

    sparse_info = sparse_info.contiguous()
    num_q_to_select = num_q_to_select.contiguous()
    sorted_q_indices = sorted_q_indices.contiguous()
    B_q, H_q, num_kv = sparse_info.shape
    # assert B == B_q and H == H_q and num_kv == NBlock_KV and NBlock_Q == NBlock_KV, "Batch and Head dimensions must match for sparse_info and sparse_kv_info"
    grid = (B, H, NBlock_Q)
    triton_fill_sparseqkv_kernel[grid](sparse_kv_info, sparse_info, 
                                       num_kv_to_select, sorted_kv_indices,
                                       num_q_to_select, sorted_q_indices,
                                          NBlock_KV, T2T=T2T)
    
    sparse_q_retio = torch.count_nonzero(sparse_info) / sparse_info.numel()
    sparse_kv_retio = torch.count_nonzero(sparse_kv_info) / sparse_kv_info.numel()
    # print(f"sparse_q_retio: {sparse_q
    # print(f"Q count_nonzero num: {torch.count_nonzero(sparse_info)}, ratio: {torch.count_nonzero(sparse_info) / sparse_info.numel()}")
    # print(f"KV count_nonzero num: {torch.count_nonzero(sparse_kv_info)}, ratio: {torch.count_nonzero(sparse_kv_info) / sparse_kv_info.numel()}")
    return sparse_kv_info, sparse_info, sparse_q_retio, sparse_kv_retio

class FluxSparseQKVAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        self.sparse_info, self.sparse_info_indptr = None, None
        self.sparse_kv_info, self.sparse_kv_info_indptr = None, None

    def sparseqkv_attn_score(self, sparse_size_for_q, sparse_size_for_kv, q, k, threshold_q = 0.05, 
                             threshold_kv = 0.15, current_iter = 0, max_sequence_length = 512,
                             sparse_info_indptr_base = None, sparse_kv_info_indptr_base=None):
        B, H, N, D = q.shape
        nblock_q = (N + sparse_size_for_q - 1) // sparse_size_for_q  # Number of blocks per feature map
        nblock_k = (N + sparse_size_for_kv - 1) // sparse_size_for_kv  # Number of blocks per feature map
        pooled_qblocks = (torch.sum(q.reshape(B, H, nblock_q, sparse_size_for_q, D),dim=3)
                          / sparse_size_for_q)
        pooled_kblocks = (torch.sum(k.reshape(B, H, nblock_k, sparse_size_for_kv, D),dim=3) 
                          / sparse_size_for_kv)

        ratio = sparse_size_for_q // sparse_size_for_kv
        pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5
        T2T = max_sequence_length // sparse_size_for_q
        T2T_kv = max_sequence_length // sparse_size_for_kv
        # B, H, N_q, N_kv
        #  T 2 T; T 2 I
        #  I 2 T, I 2 I
        # sparse_info --> for q
        t2i = pooled_score[:, :, : T2T, T2T_kv: ].contiguous()
        t2i = t2i.reshape(B, H, T2T, (N // sparse_size_for_kv - T2T_kv) // ratio, ratio)
        t2i = torch.sum(t2i, dim=[2,4]).softmax(-1)

        i2t = pooled_score[:, :, T2T:, :(N // sparse_size_for_kv - T2T_kv)].contiguous().transpose(2, 3)
        i2t = torch.sum(i2t, dim=3).softmax(-1)
        # print(f"t2i shape: {t2i.shape}, i2t shape: {i2t.shape}")
        t2i = (t2i + i2t) / 2

        sparse_info = torch.ones((B, H, nblock_q), dtype=torch.uint8, device=q.device)
        sorted_score = torch.sort(t2i, dim=-1, descending=False)
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        B, H, K = cdf.shape
        factor_ = (current_iter / 50) ** 2
        cdfthreshd = torch.full((B,), float(threshold_q * factor_), device=pooled_score.device)
        cdfthreshd_ts_q = cdfthreshd.view(B, 1, 1).expand(-1, H, 1).contiguous()
        num_to_select_q = torch.searchsorted(cdf, cdfthreshd_ts_q, right=True).squeeze(-1)
        img2_ = pooled_score.softmax(-1)
        sparse_kv_info = torch.ones((B, H, nblock_q, nblock_k), dtype=torch.uint8, device=q.device)
        sorted_kv_score = torch.sort(img2_, dim=-1, descending=False)
        cdf_kv = torch.cumsum(sorted_kv_score.values, dim=-1)
        B, H, N_blockq, N_blockkv = cdf_kv.shape
        cdfthreshd = torch.full((H,), float(threshold_kv * factor_), device=pooled_score.device)
        
        cdfthreshd_ts_kv = cdfthreshd.view(1, H, 1, 1).expand(B, -1, N_blockq, 1).contiguous()
        num_to_select_kv = torch.searchsorted(cdf_kv, cdfthreshd_ts_kv, right=True).squeeze(-1)

        sparse_kv_info, sparse_info, sparse_q_ratio, sparse_kv_ratio = fill_sparseqkv_triton(sparse_kv_info, num_to_select_kv, sorted_kv_score.indices, 
                                               sparse_info, num_to_select_q, sorted_score.indices, T2T=T2T)
        sparse_info = sparse_info.transpose(1, 2).contiguous().view(-1)
        self.sparse_info, self.sparse_info_indptr = segment_packbits(sparse_info, sparse_info_indptr_base, bitorder="little")
       
        sparse_kv_info = sparse_kv_info.transpose(1, 2).contiguous().view(-1)
        self.sparse_kv_info, self.sparse_kv_info_indptr = segment_packbits(sparse_kv_info, sparse_kv_info_indptr_base, bitorder="little")
        return sparse_q_ratio, sparse_kv_ratio
        # return 1, 1
    
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cache_dic: Optional[Dict[str, Any]] = {},
        current: Optional[Dict[str, Any]] = {},
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        head_dim = attn.to_q.weight.shape[0] // attn.heads
        
        # sparseqkv to_q
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            
        if current['type'] == 'full':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

            if current['sparseqkv'] and cache_dic['cache_index']['taylor_start'][current['stream']][current['layer']] == False:
                sparse_q_ratio, sparse_kv_ratio = self.sparseqkv_attn_score(cache_dic['sparseqkv_wrapper']._sparse_block_size_for_q,
                                cache_dic['sparseqkv_wrapper']._sparse_block_size_for_kv, 
                                query, key, cache_dic['threshold_q'], cache_dic['threshold_kv'], 
                                current['step'], cache_dic['max_sequence_length'],
                                cache_dic['sparseqkv_wrapper']._sparse_info_indptr_base, 
                                cache_dic['sparseqkv_wrapper']._sparse_kv_info_indptr_base)
                saving_sparse_info(cache_dic=cache_dic, current=current,
                                sparse_q_ratio=sparse_q_ratio, sparse_kv_ratio=sparse_kv_ratio)
                derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states.contiguous().view(-1, attn.heads, head_dim), is_attn=True)
        
        else:
            query= query.transpose(1,2).reshape(-1, attn.heads, head_dim)
            key = key.transpose(1,2).reshape(-1, attn.heads, head_dim)
            value = value.transpose(1,2).reshape(-1, attn.heads, head_dim)
            hidden_states = cache_dic['sparseqkv_wrapper'].run(query, key, value, sparse_info=self.sparse_info, 
                                        sparse_kv_info=self.sparse_kv_info, sparse_info_indptr=self.sparse_info_indptr,
                                        sparse_kv_info_indptr=self.sparse_kv_info_indptr,
                                        is_full=False, 
                                        out=taylor_formula(cache_dic=cache_dic, current=current))
                                        # out=cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0])
            
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
            
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # sparseqkv to_out
            hidden_states = attn.to_out[0](hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states