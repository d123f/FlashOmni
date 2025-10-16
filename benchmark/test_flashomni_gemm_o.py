import torch
import sparseqkv
import nvtx
from flashinfer import segment_packbits
from utils import get_qkvo_global_sparse, run_benchmark_linear

torch.manual_seed(42)
batch_size = 1
text_token = 256
only_qo_len = 32640 + text_token
qo_len = 32640 + text_token
skip_len = 0
in_dim = 3072
head_dim = 128
num_qo_heads = 24
sparse_size = 128

model_path = "./FLUX.1-dev"
from diffusers import FluxTransformer2DModel
transformer = FluxTransformer2DModel.from_pretrained(
    model_path,
    local_files_only=True,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
attn = transformer.transformer_blocks[0].attn

def mask_out1(out1, sparse_Q):
#     B, H, N / 128
    out1 = out1.view(batch_size, only_qo_len, num_qo_heads, head_dim)
    for i in range(batch_size):
        for j in range(num_qo_heads):
            for k in range(only_qo_len // sparse_size):
                if sparse_Q[i, j, k + 512 // sparse_size] == 0:
                    # print(f"masking {i}, {k}, {j}")
                    out1[i, k * sparse_size:(k + 1) * sparse_size, j, :] = 0
    return out1

def _compute_sparse_info_indptr(
    qo_indptr: torch.Tensor, num_qo_heads, sparse_block_size_for_q: int,
    device: torch.device = 'cuda'
) -> torch.Tensor:
    sparse_info_indptr = torch.empty_like(qo_indptr,device=device)
    sparse_info_indptr[0] = 0
    sparse_info_indptr[1:] = torch.cumsum(
        torch.ceil((qo_indptr[1:] - qo_indptr[:-1]) / sparse_block_size_for_q) * num_qo_heads,
        0,
    )
    sparse_info = torch.ones(sparse_info_indptr[-1], dtype=torch.uint8, device=device)
    
    return sparse_info, sparse_info_indptr
q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
sparse_info, sparse_info_indptr = _compute_sparse_info_indptr(qo_indptr=q_indptr, 
                        num_qo_heads=num_qo_heads, sparse_block_size_for_q=sparse_size)

mode = 0  # 1 for reduction
sparse_q_ratio = 0.3
sparse_info, sparse_Q, sparse_kv_info, sparse_KV = get_qkvo_global_sparse(batch_size,
                                num_qo_heads, qo_len, head_dim, spq_Q = sparse_q_ratio, spq_KV=0, mode=1, sparse_size=sparse_size)
# print(sparse_info.shape)
# print(sparse_info, sparse_info.shape)
packed_sparse_info, sparse_info_indptr = segment_packbits(sparse_info.contiguous().view(-1), sparse_info_indptr, bitorder="little")
num_repeat = 1        
with torch.no_grad():
    print("===============  after attn =============== ")
    linear = torch.nn.Linear(head_dim * num_qo_heads, in_dim, bias=False, dtype=torch.float16).cuda()
    a = torch.randn(batch_size, only_qo_len, head_dim * num_qo_heads, device="cuda", dtype=torch.float16)
    # linear = attn.to_out[0].cuda()
    b = linear.weight
    # bias = None
    bias = linear.bias
    # if bias is not None:
    #     print(bias.shape, bias.dtype)
    print(a.shape, b.shape)
    print("sparse ratio\t", 1-torch.count_nonzero(sparse_info)/sparse_info.numel())

    with nvtx.annotate("Linear_after_attn_Loop", color="blue"): # 使用上下文管理器标记范围
        for i in range(50):
            out_linear = linear(a)
        torch.cuda.synchronize()
            
    with nvtx.annotate("SparseQKV_GEMM_Reduction_Loop", color="green"):
        for i in range(50):
            out_sparseqkv_gemm_reduction = sparseqkv.sparseqkv_gemm_reduction(a, b, num_qo_heads, 
                                                        packed_sparse_info, sparse_info_indptr, skip_len, 
                                                        bias, is_for_cache=True, sparse_q_size=sparse_size)
        torch.cuda.synchronize()

    if mode == 1:
        # out_sparseqkv_gemm_reduction = torch.randn((batch_size, only_qo_len, in_dim), device="cuda", dtype=torch.float16)
        with nvtx.annotate("SparseQKV_GEMM_Reduction_2_Loop", color="green"):
            for i in range(50):
                out_sparseqkv_gemm_reduction_all = sparseqkv.sparseqkv_gemm_reduction(a, b, num_qo_heads, 
                                                                packed_sparse_info, sparse_info_indptr, skip_len, 
                                                                out_sparseqkv_gemm_reduction, is_for_cache=False,
                                                                sparse_q_size=sparse_size)
            torch.cuda.synchronize()
        diff = torch.abs(out_linear - out_sparseqkv_gemm_reduction_all)
        print(torch.allclose(out_linear, out_sparseqkv_gemm_reduction_all, atol=1e-2, rtol=1e-2))
        print(
                f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
                f"mean diff: {diff.mean().item():.6f}")
