"""CoreX/Tianshu debugger summary-record example."""

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

import flagtree.debugger as debugger
import flagtree.language as ftl

OUTPUT_DIR = Path("/tmp/flagtree_tianshu_debugger")
DEBUG_RECORD_LEVEL_VALUE = int(
    os.environ.get("FLAGTREE_DEBUGGER_RECORD_LEVEL", "1"))
DEBUG_ADDR_LEVEL_VALUE = int(
    os.environ.get("FLAGTREE_DEBUGGER_ADDR_LEVEL", "0"))
DEBUG_RECORD_LEVEL = tl.constexpr(DEBUG_RECORD_LEVEL_VALUE)
DEBUG_ADDR_LEVEL = tl.constexpr(DEBUG_ADDR_LEVEL_VALUE)
debugger.configure(
    output_dir=OUTPUT_DIR,
    record_capacity=4096,
    export_raw_records=True,
)
debugger.activate(
    level=DEBUG_RECORD_LEVEL_VALUE,
    addr_level=DEBUG_ADDR_LEVEL_VALUE,
)


@triton.jit
def debug_vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ftl.debug_collect_start(
        level=DEBUG_RECORD_LEVEL,
        addr_level=DEBUG_ADDR_LEVEL,
    )
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
    ftl.debug_collect_end()


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python-level operator used by the debugger example."""
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 256), )
    debug_vector_add_kernel[grid](x,
                                  y,
                                  out,
                                  x.numel(),
                                  BLOCK_SIZE=256,
                                  num_warps=1)
    return out


def run(n_elements: int = 4096) -> None:
    x = torch.arange(n_elements, dtype=torch.float32, device="cuda")
    y = torch.full_like(x, 2.0)
    grid = (triton.cdiv(n_elements, 256), )
    out = vector_add(x, y)
    torch.cuda.synchronize()

    expected = x + y
    runs = debugger.take_exported_runs()
    result = {
        "allclose":
        bool(torch.allclose(out, expected)),
        "device":
        torch.cuda.get_device_name(0),
        "device_index":
        torch.cuda.current_device(),
        "grid":
        list(grid),
        "max_abs_error":
        float((out - expected).abs().max().item()),
        "exported_runs":
        len(runs),
        "reports": [run.get("report_path") for run in runs],
        "decoded_headers":
        [run.get("decoded", {}).get("header", {}) for run in runs],
        "n_elements":
        n_elements,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
