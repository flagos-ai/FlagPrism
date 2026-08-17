import flagtree.profiler as profiler

import torch
import triton
import sys

from helper_kernels import custom_add, matmul_kernel


def torch_device():
    backend = triton.runtime.driver.active.get_current_target().backend
    return "npu" if backend in {"ascend", "npu"} else "cuda"


def main():
    a = torch.zeros(1, device=torch_device())
    with profiler.scope("test"):
        custom_add[(1, )](a)


def test_main():
    main()


def matmul():
    device = torch_device()
    a = torch.randn((32, 32), device=device, dtype=torch.float16)
    b = torch.randn((32, 32), device=device, dtype=torch.float16)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    matmul_kernel[(1, )](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        128,
        256,
        64,
        8)
    return c


if __name__ == "__main__":
    if sys.argv[1] == "test":
        main()
    elif sys.argv[1] == "test_matmul":
        matmul()
