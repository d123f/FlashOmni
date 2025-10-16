import torch
import sparseqkv
import nvtx
from flashinfer import segment_packbits
from utils import get_qkvo_global_sparse, run_benchmark_linear

batch_size = 1
only_qo_len = 4096
qo_len = only_qo_len + 512
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

sparse_info, sparse_Q, sparse_kv_info, sparse_KV = get_qkvo_global_sparse(batch_size,
                                num_qo_heads, qo_len, head_dim, spq_Q = 0.5, spq_KV=0, mode=1, sparse_size=sparse_size)
# print(sparse_info.shape)
# print(sparse_info, sparse_info.shape)
packed_sparse_info, sparse_info_indptr = segment_packbits(sparse_info.contiguous().view(-1), sparse_info_indptr, bitorder="little")
num_repeat = 1        

with torch.no_grad():
    # =========  before attn
    print("===============  before attn =============== ")
    # linear = torch.nn.Linear(in_dim, head_dim * num_qo_heads, bias=True, dtype=torch.bfloat16).cuda()
    linear = attn.to_q.cuda()
    b = linear.weight
    a = torch.randn(batch_size, only_qo_len, in_dim, device="cuda", dtype=torch.bfloat16)

    # bias = None
    bias = linear.bias

    with nvtx.annotate("Linear_Loop", color="blue"): # 使用上下文管理器标记范围
        for i in range(num_repeat):
            out_linear = linear(a)
    out1 = mask_out1(out_linear, sparse_Q).contiguous()
    
    with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
        for i in range(num_repeat):
            out_sparseqkv_linear = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads, packed_sparse_info, 
                                                        sparse_info_indptr, 512, bias, sparse_q_size=sparse_size)
    out = out_sparseqkv_linear.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()

    diff = torch.abs(out - out1)
    print(torch.allclose(out, out1, atol=1e-2, rtol=1e-2))
    print(
            f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
            f"mean diff: {diff.mean().item():.6f}")
    
    with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
        for i in range(num_repeat):
            out_sparseqkv_linear_full = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads, packed_sparse_info, 
                                                        sparse_info_indptr, 512, bias, sparse_q_size=sparse_size,
                                                        is_full=False)
    # out = out_sparseqkv_linear_full.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()
    out_linear = linear(a)
    diff = torch.abs(out_sparseqkv_linear_full - out_linear)
    print(torch.allclose(out_sparseqkv_linear_full, out_linear, atol=1e-2, rtol=1e-2))
    print(
            f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
            f"mean diff: {diff.mean().item():.6f}")
    
    with nvtx.annotate("SparseQKV_GEMM_Loop", color="green"):
        for i in range(num_repeat):
            out_sparseqkv_linear_full = sparseqkv.sparseqkv_gemm(a, b, num_qo_heads,  bias=bias,
                                                        is_full=True)
    # out = out_sparseqkv_linear_full.view(batch_size, only_qo_len, num_qo_heads, head_dim).contiguous()
    out_linear = linear(a)
    diff = torch.abs(out_sparseqkv_linear_full - out_linear)
    print(torch.allclose(out_sparseqkv_linear_full, out_linear, atol=1e-2, rtol=1e-2))
    print(
            f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
            f"mean diff: {diff.mean().item():.6f}")
    # print(out)
    # print(out1)
    # # =========  after attn
    # print("===============  after attn =============== ")
    # linear = torch.nn.Linear(head_dim * num_qo_heads, in_dim, bias=False, dtype=torch.float16).cuda()
    # a = torch.randn(batch_size, only_qo_len, head_dim * num_qo_heads, device="cuda", dtype=torch.float16)
    # # linear = attn.to_out[0].cuda()
    # b = linear.weight
    # # bias = None
    # bias = linear.bias
    # # if bias is not None:
    # #     print(bias.shape, bias.dtype)

    # with nvtx.annotate("Linear_after_attn_Loop", color="blue"): # 使用上下文管理器标记范围
    #     for i in range(10):
    #         out_linear = linear(a)
    #     torch.cuda.synchronize()
            
    # with nvtx.annotate("SparseQKV_GEMM_Reduction_Loop", color="green"):
    #     for i in range(10):
    #         out_sparseqkv_gemm_reduction = sparseqkv.sparseqkv_gemm_reduction(a, b, num_qo_heads, 
    #                                                     packed_sparse_info, sparse_info_indptr, 512, 
    #                                                     bias, is_for_cache=True, sparse_q_size=sparse_size)
    #     torch.cuda.synchronize()

    # # out_sparseqkv_gemm_reduction = torch.randn((batch_size, only_qo_len, in_dim), device="cuda", dtype=torch.float16)
    # with nvtx.annotate("SparseQKV_GEMM_Reduction_2_Loop", color="green"):
    #     for i in range(10):
    #         out_sparseqkv_gemm_reduction_all = sparseqkv.sparseqkv_gemm_reduction(a, b, num_qo_heads, 
    #                                                         packed_sparse_info, sparse_info_indptr, 512, 
    #                                                         out_sparseqkv_gemm_reduction, is_for_cache=False,
    #                                                         sparse_q_size=sparse_size)
    #     torch.cuda.synchronize()
    # diff = torch.abs(out_linear - out_sparseqkv_gemm_reduction_all)
    # print(torch.allclose(out_linear, out_sparseqkv_gemm_reduction_all, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")


    # # ==================  split gemm 
    # print("split attn")
    # linear = torch.nn.Linear(head_dim * num_qo_heads, in_dim, bias=False, dtype=torch.float16).cuda()
    # a = torch.randn((batch_size, only_qo_len, num_qo_heads * head_dim), device="cuda", dtype=torch.float16)
    # # linear = attn.to_out[0].cuda()
    # b = linear.weight
    # bias = linear.bias
    # if bias is not None:
    #     print(bias.shape, bias.dtype)

    # with nvtx.annotate("Linear_split_attn_Loop", color="blue"): # 使用上下文管理器标记范围
    #     for i in range(10):
    #         out_linear = linear(a)
    #     torch.cuda.synchronize()
        
    # # out_sparseqkv_gemm_reduction = torch.randn((batch_size, only_qo_len, in_dim), device="cuda", dtype=torch.float16)
    # with nvtx.annotate("SparseQKV_GEMM_split_2_Loop", color="green"):
    #     for i in range(4):
    #         out_sparseqkv_gemm_split, cached_hidden = sparseqkv.sparseqkv_gemm_split(a, b, num_qo_heads, 
    #                                                         packed_sparse_info, sparse_info_indptr,  512, bias)
    #         torch.cuda.synchronize()
    
    # print("out_linear", out_linear)
    # print("out_sparseqkv_gemm_split", out_sparseqkv_gemm_split)
    # print("cached_hidden", cached_hidden)
    # diff = torch.abs(out_linear - out_sparseqkv_gemm_split)
    # print(torch.allclose(out_linear, out_sparseqkv_gemm_split, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")
    
    # diff = torch.abs(out_linear - cached_hidden)
    # print(torch.allclose(out_linear, cached_hidden, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")


    # # =============== after attn quant int8 for SM80 w8a8
    # print("=============== after attn quant int8 for SM80 w8a8 ================")
    # input = torch.randint(-80, 80, (batch_size, only_qo_len, in_dim), dtype=torch.int8).to("cuda")
    # weight = torch.randint(-80, 80, (head_dim * num_qo_heads, in_dim), dtype=torch.int8).to("cuda")
    # zp_weight = torch.randint(-10, 10, (head_dim * num_qo_heads,), dtype=torch.int16).to("cuda")

    # scale_input = 0.01 * torch.rand(batch_size, only_qo_len, dtype=torch.float16).to("cuda") + 0.005
    # scale_weight = 0.1 * torch.rand(head_dim * num_qo_heads, dtype=torch.float16).to("cuda") + 0.1
    # # bias = torch.tensor([0.0 for _ in range(N)], dtype=torch.float16).to("cuda")
    # bias = torch.rand(num_qo_heads * head_dim, dtype=torch.float16).to("cuda") * 200

    # # sum the token dim
    # input_sum = (scale_input.view(batch_size, only_qo_len, 1).to(torch.float32) * input.to(torch.float32)).sum(dim=2).to(torch.float16)

    # input_fp16 = input.to(torch.float16)
    # input_fp32 = input.to(torch.float32)
    # weight_fp16 = weight.to(torch.float16)
    # weight_fp32 = weight.to(torch.float32)
    # for i in range(10):
    #     output_gt = (torch.nn.functional.linear(input_fp32, weight_fp32) * scale_input.view(batch_size, only_qo_len, 1).to(torch.float32) * scale_weight.view(1, -1).to(torch.float32)
    #             + input_sum.view(batch_size, only_qo_len, 1).to(torch.float32) * zp_weight.to(torch.float32).view(1, -1) * scale_weight.view(1, -1).to(torch.float32) 
    #             + bias.to(torch.float32)).to(torch.float16)
    
    # for i in range(10):
    #     output_gt = input_fp16 @ weight_fp16.T + bias.to(torch.float16)
        
    # # print(output_gt)
    # # exit()
    # for i in range(10):
    #     output = sparseqkv.sparseqkv_gemm_int8(input, weight, scale_input, scale_weight, input_sum, zp_weight, bias).view(batch_size, only_qo_len, num_qo_heads, head_dim)
    # torch.cuda.synchronize()
    # output_gt_normal = output_gt.contiguous().view(batch_size, only_qo_len, num_qo_heads, head_dim)
    # diff = torch.abs(output_gt_normal - output)
    # print(torch.allclose(output_gt_normal, output, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")
    

    # output_gt_mask = mask_out1(output_gt, sparse_Q)

    # torch.cuda.synchronize()
    # for i in range(10):
    #     output = sparseqkv.sparseqkv_gemm_int8(input, weight, scale_input, scale_weight, input_sum, zp_weight, bias,
    #                                         num_qo_heads, packed_sparse_info, sparse_info_indptr, 512).view(batch_size, only_qo_len, num_qo_heads, head_dim)
    # torch.cuda.synchronize()
    # diff = torch.abs(output_gt_mask - output)
    # print(torch.allclose(output_gt_mask, output, atol=1e-2, rtol=1e-2))
    # print(
    #         f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")
    

    # # =============== after attn quant int8 for SM80 w8a16
    # print("after attn quant")
    # a = torch.randn(batch_size, qo_len, num_qo_heads * head_dim, device="cuda", dtype=torch.float16)
    # b = torch.randint(0, 128, (in_dim, num_qo_heads * head_dim), device="cuda", dtype=torch.uint8)
    # # b = torch.full((in_dim, num_qo_heads * head_dim), fill_value=129, device="cuda", dtype=torch.uint8)

    # b_l = b.reshape(in_dim, num_qo_heads * head_dim // 16, 8, 2)[:,:,:4,:]
    # b_r = b.reshape(in_dim, num_qo_heads * head_dim // 16, 8, 2)[:,:,4:,:]
    # b_cat = torch.cat((b_l, b_r), dim=-1).reshape(in_dim, num_qo_heads * head_dim).contiguous()


    # b_fp16 = b.to(torch.float16) - 128
    # # linear = attn.to_out[0].cuda()
    # # a = torch.randint(-127, 128, (batch_size, only_qo_len, head_dim * num_qo_heads), device="cuda", dtype=torch.float16)
    # # bias = linear.bias
    # # if bias is not None:
    # #     print(bias.shape, bias.dtype)

    # with nvtx.annotate("Linear_after_attn_Loop", color="blue"): # 使用上下文管理器标记范围
    #     for i in range(10):
    #         # print(a.shape)
    #         # print(b_fp16.T.shape)
    #         out_linear = a @ b_fp16.T
    #     torch.cuda.synchronize()

    # out_sparseqkv_gemm_reduction = torch.randn((batch_size, only_qo_len, in_dim), device="cuda", dtype=torch.float16)
    # with nvtx.annotate("SparseQKV_GEMM_Reduction_2_Loop", color="green"):
    #     for i in range(10):
    #         out_sparseqkv_gemm_reduction_all = sparseqkv.sparseqkv_gemm_reduction_quant(a, b_cat, num_qo_heads, 
    #                                                         packed_sparse_info, sparse_info_indptr, 0, 
    #                                                         bias=None, is_for_cache=True)
    #     torch.cuda.synchronize()

    # with nvtx.annotate("SparseQKV_GEMM_Reduction_2_Loop2", color="green"):
    #     for i in range(10):
    #         out_sparseqkv_gemm_reduction_all_2 = sparseqkv.sparseqkv_gemm_reduction_quant(a, b_cat, num_qo_heads, 
    #                                                         packed_sparse_info, sparse_info_indptr, 0, 
    #                                                         bias=out_sparseqkv_gemm_reduction_all, is_for_cache=False)
    #     torch.cuda.synchronize()

    # print(out_sparseqkv_gemm_reduction_all_2)
    # print(out_linear)
    # diff = torch.abs(out_linear - out_sparseqkv_gemm_reduction_all_2)
    # print(torch.allclose(out_linear, out_sparseqkv_gemm_reduction_all_2, atol=1e-2, rtol=1e-2))
    # print(f"max diff: {diff.max().item():.6f}, min diff: {diff.min().item():.6f}, " 
    #         f"mean diff: {diff.mean().item():.6f}")
    