import torch
import torch.nn as nn
from sparseqkv.gemm import sparseqkv_gemm_int8, sparseqkv_gemm_to_q_int8
from base import QuantParams

class SparseQKVW8A8OF16DynaLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_sym: bool = True,
        is_to_q: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.is_to_q = is_to_q

        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "bias",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ) if self.has_bias else None,
        )
        
        self.register_buffer(
            "scale_weight",
            torch.empty(
                self.out_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "zp_weight",
            torch.empty(
                self.out_features,
                dtype=torch.int16,
                requires_grad=False,
            ) if not weight_sym else None,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_sym: bool = True,
        init_only: bool = False,
        is_to_q: bool = False,
    ):
        q_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            has_bias=linear.bias is not None,
            weight_sym=weight_sym,
            is_to_q=is_to_q,
        )

        assert linear.weight.dtype == torch.float16
    
        if init_only:
            return q_linear

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)

        ### quantize the weight ###
        fp16_weight = linear.weight.data

        if weight_sym:
            weight_scale = fp16_weight.abs().max(dim=-1).values / 127.0
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)), -128, 127)

            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()
            
        else:
            weight_max = fp16_weight.max(dim=-1).values
            weight_min = fp16_weight.min(dim=-1).values

            weight_scale = (weight_max - weight_min) / 255.0
            weight_zp = torch.round((weight_min) / weight_scale) + 128
            weight_quant = torch.clamp(torch.round(fp16_weight / weight_scale.view(-1, 1)) - weight_zp.view(-1, 1), -128, 127)

            q_linear.zp_weight.data[:] = weight_zp.to(torch.int16).contiguous()
            q_linear.weight.data[:, :] = weight_quant.to(torch.int8).contiguous()
            q_linear.scale_weight.data[:] = weight_scale.to(torch.float16).contiguous()

        return q_linear
    
    def forward(self,
        input: torch.Tensor,
        quant_params: QuantParams,
        num_qo_heads: int = None,
        sparse_info: torch.Tensor = None, 
        sparse_info_indptr: torch.Tensor = None,
        text_len: int = None,
    ):
        if self.is_to_q:
            output = sparseqkv_gemm_to_q_int8(
                input,
                self.weight,
                quant_params.scale_input,
                self.scale_weight,
                quant_params.sum_input,
                self.zp_weight,
                num_qo_heads,
                sparse_info,
                sparse_info_indptr,
                text_len,
                self.bias,
            )
        else:
            output = sparseqkv_gemm_int8(
                input,
                self.weight,
                quant_params.scale_input,
                self.scale_weight,
                quant_params.sum_input,
                self.zp_weight,
                self.bias,
            )
        return output
    
if __name__ == "__main__":
    # Example usage
    batch_size = 1
    seq_len = 4096
    in_dim = 3072
    out_dim = 3072
    linear = nn.Linear(in_dim, out_dim, bias=True, dtype=torch.float16, device="cuda")
    q_linear = SparseQKVW8A8OF16DynaLinear.from_linear(linear, weight_sym=False)


    input_tensor = torch.randint(-80, 80, (batch_size, seq_len, in_dim), dtype=torch.int8).to("cuda")
    quant_params = QuantParams(batch_size, seq_len, has_sum_input=True)
    quant_params.scale_input.fill_(0.01)
    quant_params.sum_input.fill_(0.0)

    output = q_linear(input_tensor, quant_params)
    print(output.shape)  # Should be [32, 64]
    print(output)

    input_fp16 = input_tensor.contiguous().to(torch.float16)
    output_gt = (linear(input_fp16) * quant_params.scale_input.view(batch_size, seq_len, 1).to(torch.float16) * q_linear.scale_weight.view(1, -1).to(torch.float16)
                + quant_params.sum_input.view(batch_size, seq_len, 1).to(torch.float16) * q_linear.zp_weight.to(torch.float16).view(1, -1) * q_linear.scale_weight.view(1, -1).to(torch.float16) 
                + q_linear.bias.to(torch.float16)).to(torch.float16)
        
    print(output.shape, output_gt.shape)
    print(torch.allclose(output, output_gt, atol=1e-3))
