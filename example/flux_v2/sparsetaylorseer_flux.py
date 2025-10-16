from diffusers import DiffusionPipeline
from diffusers.utils import logging
import torch
import os, gc
from models.forwards import (sparsetaylorseer_flux_forward, 
                        sparsetaylorseer_flux_single_block_forward,
                        sparsetaylorseer_flux_double_block_forward)

from models.sparseqkv_attn_processor.modify_flux import set_sparseqkv_attn_flux

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 0
file_path = "./coco_prompts_val2017_noidex.txt"
skip = 0
with open(file_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()[skip:1]
pipeline = DiffusionPipeline.from_pretrained("./FLUX.1-dev", torch_dtype=torch.bfloat16)

# TaylorSeer settings
pipeline.transformer.__class__.num_steps = num_inference_steps

pipeline.transformer.__class__.forward = sparsetaylorseer_flux_forward

set_sparseqkv_attn_flux(
    pipeline.transformer,
    batch_size=1,
    img_len=4096,
    num_text_tokens=512,
    sparse_block_size_for_q=128,
    sparse_block_size_for_kv=128,
)

for double_transformer_block in pipeline.transformer.transformer_blocks:
    double_transformer_block.__class__.forward = sparsetaylorseer_flux_double_block_forward
    
for single_transformer_block in pipeline.transformer.single_transformer_blocks:
    single_transformer_block.__class__.forward = sparsetaylorseer_flux_single_block_forward

pipeline.to("cuda")

parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")

torch.cuda.reset_peak_memory_stats()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out_path = './1024_1024_wlaststep/50_15_0.3_O1_N6_W9'
os.makedirs(out_path, exist_ok=True)
for i, prompt in enumerate(prompts):
    img = pipeline(
        prompt.strip(), 
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator(device="cuda").manual_seed(0)
        ).images[0]
    img.save(f"{out_path}/output_{i+skip}.jpg")
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end) * 1e-3
peak_memory = torch.cuda.max_memory_allocated(device="cuda")


print(
    f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
)