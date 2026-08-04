import triton
import argparse
import ctypes
import flagtree.profiler as profiler
import flagtree.profiler.language as pl
from flagtree.profiler.hooks import InstrumentationHook

pl.enable_semantic("triton")


def write_tensor_to_file(tensor, filename):
    data_ptr = tensor.data_ptr()
    size = tensor.numel()
    dtype_size = tensor.element_size()
    total_bytes = size * dtype_size

    with open(filename, 'wb') as f:
        data_arr = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_ubyte * total_bytes))
        f.write(bytes(data_arr.contents))


@triton.jit
def seq_kernel():
    pl.enter_scope("r0")
    pl.enter_scope("r1")
    pl.enter_scope("r2")
    pl.exit_scope("r1")
    pl.exit_scope("r0")
    pl.exit_scope("r2")


def seq(args):
    grid_size = 2
    grid = (grid_size, )
    profiler.start("", backend="instrumentation", mode=profiler.mode.Default(buffer_size=256))
    InstrumentationHook.enable_host_buffer = True
    InstrumentationHook.profile_buffer_size = 512
    seq_kernel[grid]()
    write_tensor_to_file(InstrumentationHook.host_buffer, args.trace_file)
    profiler.finalize()


@triton.jit
def loop_kernel():
    for k in range(0, 20):
        pl.enter_scope("r0")
        pl.exit_scope("r0")


def loop(args):
    grid_size = 1
    grid = (grid_size, )
    profiler.start("", backend="instrumentation", mode=profiler.mode.Default(buffer_size=256))
    InstrumentationHook.enable_host_buffer = True
    InstrumentationHook.profile_buffer_size = 512
    loop_kernel[grid]()
    write_tensor_to_file(InstrumentationHook.host_buffer, args.trace_file)
    profiler.finalize()


def main():
    parser = argparse.ArgumentParser(description='FlagTree Profiler intra-kernel trace generator')
    parser.add_argument('trace_file', type=str, help='Trace file path')
    parser.add_argument('--kernel', '-k', type=str, help='Kernel name')
    args = parser.parse_args()

    if args.kernel == "seq":
        seq(args)
    if args.kernel == "loop":
        loop(args)


if __name__ == '__main__':
    main()
