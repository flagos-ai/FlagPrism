"""Run one Python-level operator on Tianshu/CoreX for profiler collection."""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python operator backed by a Tianshu Triton kernel."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 256
    _vector_add_kernel[(triton.cdiv(n_elements, block_size),)](
        x, y, out, n_elements, BLOCK_SIZE=block_size
    )
    return out


def run(n_elements: int = 4096) -> None:
    x = torch.arange(n_elements, dtype=torch.float32, device="cuda")
    y = torch.full_like(x, 2.0)
    out = vector_add(x, y)
    torch.cuda.synchronize()
    expected = x + y
    print(json.dumps({
        "allclose": bool(torch.allclose(out, expected)),
        "device": torch.cuda.get_device_name(0),
        "max_abs_error": float((out - expected).abs().max().item()),
        "n_elements": n_elements,
        "operator": "vector_add",
    }, sort_keys=True))


if __name__ == "__main__":
    run()
