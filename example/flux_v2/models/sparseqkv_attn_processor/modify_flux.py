from diffusers import FluxTransformer2DModel
from .attention_processor import FluxSparseQKVAttnProcessor
import torch
import sparseqkv

def set_sparseqkv_attn_flux(
    model: FluxTransformer2DModel,
    verbose=False,
    batch_size: int = 1,
    img_len: int = 4096,
    num_text_tokens = 512,
    sparse_block_size_for_q: int = 128,
    sparse_block_size_for_kv: int = 128,
):
    print("Set sparseqkv sparseqkv_attention kernel wrapper")
    head_dim = model.transformer_blocks[0].attn.inner_dim // model.transformer_blocks[0].attn.heads
    model.sparseqkv_wrapper = sparseqkv_kernel(
        batch_size, 
        img_len + num_text_tokens, 
        model.transformer_blocks[0].attn.heads, 
        head_dim,
        sparse_block_size_for_q = sparse_block_size_for_q,
        sparse_block_size_for_kv = sparse_block_size_for_kv,
        dtype = model.transformer_blocks[0].attn.to_q.weight.dtype,
    )
    print("Set sparseqkv attention processor")
    for block in model.transformer_blocks:
        block.attn.verbose = verbose
        processor = FluxSparseQKVAttnProcessor()
        block.attn.set_processor(processor)

    for block in model.single_transformer_blocks:
        block.attn.verbose = verbose
        processor = FluxSparseQKVAttnProcessor()
        block.attn.set_processor(processor)


def sparseqkv_kernel(
    batch_size, 
    qo_len, 
    num_qo_heads, 
    head_dim, 
    sparse_block_size_for_q: int = 128,
    sparse_block_size_for_kv: int = 128,
    causal=False, 
    pos_encoding_mode='NONE', 
    logits_soft_cap=0.0,
    dtype=torch.bfloat16,
):
    kv_len = qo_len
    num_kv_heads = num_qo_heads
    kv_layout = "NHD"
    
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    # workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    workspace_buffer = torch.empty(256, dtype=torch.int8, device="cuda:0")
    wrapper = sparseqkv.mySparseFA.BatchSparseFAWithRaggedKVWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sparse_block_size_for_q,
        sparse_block_size_for_kv,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        q_data_type=dtype
        # use_fp16_qk_reduction=True,
    )
    return wrapper
