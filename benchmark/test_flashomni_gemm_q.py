import torch
import sparseqkv
import nvtx
from flashinfer import segment_packbits
from utils import get_qkvo_global_sparse, run_benchmark_linear

torch.manual_seed(42)
batch_size = 1
only_qo_len = 4096+ 512
qo_len = 4096 + 512
in_dim = 3072
head_dim = 128
num_qo_heads = 24
sparse_size = 128
text_skip = 0

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
                if sparse_Q[i, j, k + text_skip// sparse_size] == 0:
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

sparse_info, sparse_Q, sparse_kv_info, sparse_KV = get_qkvo_global_sparse(batch_size,
                                num_qo_heads, qo_len, head_dim, spq_Q = 0.5, spq_KV=0, mode=1, sparse_size=sparse_size)
# print(sparse_info.shape)
# print(sparse_info, sparse_info.shape)
packed_sparse_info, sparse_info_indptr = segment_packbits(sparse_info.contiguous().view(-1), sparse_info_indptr, bitorder="little")
num_repeat = 50        
warmup = 5
with torch.no_grad():
    # =========  before attn
    print("===============  before attn =============== ")
    print("===============  full linear with mask vs gemm q=============== ")
    # linear = torch.nn.Linear(in_dim, head_dim * num_qo_heads, bias=True, dtype=torch.bfloat16).cuda()
    linear = attn.to_q.cuda()
    b = linear.weight
    a = torch.randn(batch_size, only_qo_len, in_dim, device="cuda", dtype=torch.bfloat16)

    # bias = None
    bias = linear.bias
    print(a.shape, b.shape)
    print("sparse ratio\t", 1-torch.count_nonzero(sparse_info)/sparse_info.numel())
    for i in range(warmup):
        _ = linear(a)
    with nvtx.annotate("Linear_Loop", color="blue"): # 使用上下文管理器标记范围
        for i in range(num_repeat):
            out_linear = linear(a)
    out1 = mask_out1(out_linear, sparse_Q).contiguous()
    
    for i in range(warmup):
        _ = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads, packed_sparse_info, 
                        sparse_info_indptr, text_skip, bias, sparse_q_size=sparse_size)
    with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
        for i in range(num_repeat):
            out_sparseqkv_linear = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads, packed_sparse_info, 
                                                        sparse_info_indptr, text_skip, bias, sparse_q_size=sparse_size)
    out = out_sparseqkv_linear.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()

    diff = torch.abs(out - out1)
    print(torch.allclose(out, out1, atol=1e-2, rtol=1e-2))
    print(
            f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
            f"mean diff: {diff.mean().item():.6f}")
    
    # print("===============  full linear vs gemm q with full=============== ")
    
    # with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
    #     for i in range(num_repeat):
    #         out_sparseqkv_linear_full = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads, packed_sparse_info, 
    #                                                     sparse_info_indptr, 512, bias, sparse_q_size=sparse_size,
    #                                                     is_full=False)
    # # out = out_sparseqkv_linear_full.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()
    # out_linear = linear(a)
    # diff = torch.abs(out_sparseqkv_linear_full - out_linear)
    # print(torch.allclose(out_sparseqkv_linear_full, out_linear, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")
    
    # with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
    #     for i in range(num_repeat):
    #         out_sparseqkv_linear_full = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads,  bias=bias,
    #                                                     is_full=True)
    # # out = out_sparseqkv_linear_full.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()
    # out_linear = linear(a)
    # diff = torch.abs(out_sparseqkv_linear_full - out_linear)
    # print(torch.allclose(out_sparseqkv_linear_full, out_linear, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")