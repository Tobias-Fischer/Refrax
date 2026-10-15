"""Inference time of Depth Anything V2 Small (the depth source assumed for real-time
depth-aware refraction removal), via transformers.

Usage:
    python benchmarks/bench_depth_anything.py [--device mps] [--repeat 20] [--pretrained]

By default the network is built from the Depth-Anything-V2-Small-hf architecture
(DINOv2 ViT-S/14 backbone + DPT head, ~25 M parameters) with random weights: inference
time does not depend on the weight values and nothing is downloaded. --pretrained loads
depth-anything/Depth-Anything-V2-Small-hf from the Hugging Face hub instead.
Input sizes: 518 x 518 and 518 x 924 (the processor's default for a 16:9 frame: shorter
side 518, multiples of 14), fp32 and fp16, batch 1, plus the bilinear upsampling of the
prediction to 1920 x 1080 (what Corrector does to a low-resolution depth).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.torch_remove import default_device, synchronize  # noqa: E402


def build(pretrained):
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config
    if pretrained:
        return DepthAnythingForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf")
    backbone = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                            intermediate_size=1536, patch_size=14, image_size=518,
                            out_features=["stage3", "stage6", "stage9", "stage12"],
                            reshape_hidden_states=False, apply_layernorm=True)
    cfg = DepthAnythingConfig(backbone_config=backbone, reassemble_factors=[4, 2, 1, 0.5],
                              neck_hidden_sizes=[48, 96, 192, 384], fusion_hidden_size=64,
                              head_hidden_size=32, depth_estimation_type="relative")
    return DepthAnythingForDepthEstimation(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()
    dev = args.device
    model = build(args.pretrained).eval().to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"Depth Anything V2 Small ({'pretrained' if args.pretrained else 'random weights'}), "
          f"{n / 1e6:.1f} M parameters, device {dev}, torch {torch.__version__}, "
          f"load {os.getloadavg()[0]:.1f}")
    dtypes = [torch.float32] + ([torch.float16] if torch.device(dev).type != "cpu" else [])
    for dtype in dtypes:
        m = model.to(dtype)
        if dtype == torch.float16 and torch.device(dev).type == "mps":
            # torch 2.14's fused fp16 SDPA kernel fails to build on this Mac: plain attention.
            m.config._attn_implementation = "eager"
            m.config.backbone_config._attn_implementation = "eager"
            for mod in m.modules():
                if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                    mod.config._attn_implementation = "eager"
        for h, w in ((518, 518), (518, 924)):
            x = torch.randn(1, 3, h, w, device=dev, dtype=dtype)
            ts = []
            with torch.inference_mode():
                for i in range(args.warmup + args.repeat):
                    synchronize(dev)
                    t = time.perf_counter()
                    d = m(pixel_values=x).predicted_depth
                    up = F.interpolate(d[:, None].float(), size=(1080, 1920), mode="bilinear",
                                       align_corners=False)
                    synchronize(dev)
                    if i >= args.warmup:
                        ts.append(1e3 * (time.perf_counter() - t))
            ts = torch.tensor(ts)
            print(f"{str(dtype).split('.')[-1]:>8} {h}x{w}: median {ts.median():.1f} ms "
                  f"(p10 {ts.quantile(0.1):.1f}, p90 {ts.quantile(0.9):.1f}), output "
                  f"{tuple(d.shape[-2:])} -> {tuple(up.shape[-2:])}")


if __name__ == "__main__":
    main()
