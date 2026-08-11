"""Minimal CoreX/Tianshu vector-add operator used by FlagPrism demos."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def run(n_elements: int = 4096):
    device = torch.device("cuda")
    x = torch.arange(n_elements, device=device, dtype=torch.float32)
    y = torch.full_like(x, 2.0)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, 256),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256, num_warps=1)
    torch.cuda.synchronize()

    expected = x + y
    result = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_index": torch.cuda.current_device(),
        "n_elements": n_elements,
        "grid": list(grid),
        "max_abs_error": float((out - expected).abs().max().item()),
        "allclose": bool(torch.allclose(out, expected)),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["allclose"]:
        raise RuntimeError(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-elements", type=int, default=4096)
    args = parser.parse_args()
    run(args.n_elements)
