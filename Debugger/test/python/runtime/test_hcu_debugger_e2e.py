# FlagPrism：verify HIP hidden arguments, stream ordering and tensor payload contents.
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
import flagtree.language as ftl
from flagtree import debugger


@pytest.fixture(autouse=True)
def hcu_device():
    if not torch.cuda.is_available():
        pytest.skip("HIP device required")
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU target required")
    assert debugger.is_available(), "HCU debugger bindings unavailable"
    debugger.clear_exported_runs()
    debugger.reset_config()
    yield
    debugger.deactivate()
    debugger.reset_config()


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_hcu_level2_value_and_address_contents(tmp_path, dtype):

    @triton.jit
    def capture_copy(X, Y):
        offset = tl.arange(0, 16)
        ftl.debug_collect_start(level=2, addr_level=2)
        value = tl.load(X + offset)
        tl.store(Y + offset, value)
        ftl.debug_collect_end()

    x = torch.arange(-8, 8, device="cuda", dtype=dtype) + 0.25
    y = torch.empty_like(x)
    debugger.activate(level=2,
                      addr_level=2,
                      timeline=False,
                      output_dir=tmp_path)
    capture_copy[(1, )](x, y, num_warps=1)
    torch.cuda.synchronize()
    runs = debugger.take_exported_runs()
    assert torch.equal(x, y)
    assert len(runs) == 1
    assert runs[0]["meta"]["backend_kind"] == 2
    assert runs[0]["decoded"]["header"]["overflow_count"] == 0
    artifacts = runs[0]["runtime_metadata"]["full_dump_artifacts"]
    values = [a for a in artifacts if a["kind"] == "value"]
    addresses = [a for a in artifacts if a["kind"] == "memory_address"]
    assert values and addresses
    expected = x.float().cpu().numpy()
    for artifact in values:
        assert Path(artifact["path"]).is_file()
        tensor = np.load(artifact["path"], allow_pickle=False)
        # Pointer arithmetic may also be observed; compare all float value dumps.
        if tensor.dtype == np.float32:
            np.testing.assert_array_equal(tensor.reshape(-1), expected)
    assert any(
        np.load(a["path"], allow_pickle=False).dtype == np.float32
        for a in values)
    bases = (x.data_ptr(), y.data_ptr())
    for artifact in addresses:
        actual = np.load(artifact["path"], allow_pickle=False).reshape(-1)
        contiguous = any(
            np.array_equal(
                actual, base +
                np.arange(16, dtype=np.uint64) * x.element_size())
            for base in bases)
        # A tt.splat pointer result is the broadcast base, before tt.addptr.
        broadcast = any(
            np.array_equal(actual, np.full(16, base, dtype=np.uint64))
            for base in bases)
        assert contiguous or (artifact["source"] == "result" and broadcast), (
            "Captured pointer result or memory-access addresses are incorrect")
    accesses = [a for a in addresses if a["source"] == "address"]
    assert len(
        accesses
    ) == 2, "Both the load and store must expose actual lane addresses"


def test_hcu_masked_non_default_stream_launches():

    @triton.jit
    def masked_copy(X, Y, N: tl.constexpr):
        offset = tl.program_id(0) * 32 + tl.arange(0, 32)
        ftl.debug_collect_start(level=1, addr_level=1)
        value = tl.load(X + offset, offset < N, other=0.0)
        tl.store(Y + offset, value, offset < N)
        ftl.debug_collect_end()

    n = 37
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    outputs = [torch.empty_like(x), torch.empty_like(x)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    ready = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(ready)
    debugger.activate(level=1, addr_level=1, timeline=False, output_dir=None)
    for stream, output in zip(streams, outputs):
        with torch.cuda.stream(stream):
            masked_copy[(2, )](x, output, n, num_warps=1)
    for stream in streams:
        stream.synchronize()
    runs = debugger.take_exported_runs()
    assert all(torch.equal(x, output) for output in outputs)
    assert len(runs) == 2
    assert all(run["meta"]["backend_kind"] == 2 for run in runs)
    assert all(run["decoded"]["records"] for run in runs)
    assert all(run["decoded"]["header"]["overflow_count"] == 0 for run in runs)


# FlagPrism end


# FlagPrism：exercise summary boundaries, overflow and disabled-launch ABI on HIP.
@triton.jit
def hcu_summary_copy(X, Y):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    ftl.debug_collect_start(level=1)
    value = tl.load(X + i)
    tl.store(Y + i, value)
    ftl.debug_collect_end()


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("all_nonfinite", [False, True])
def test_hcu_nonfinite_summary(dtype, all_nonfinite):
    data = (
        [float("nan"), float("inf"), -float("inf"),
         float("nan")] * 4 if all_nonfinite else
        [float("nan"),
         float("inf"), -float("inf"), 0.0, -0.0, 1.0, -2.0, 3.0] * 2)
    x = torch.tensor(data, device="cuda", dtype=dtype)
    y = torch.empty_like(x)
    debugger.activate(level=1, timeline=False, output_dir=None)
    hcu_summary_copy[(1, )](x, y, num_warps=1)
    torch.testing.assert_close(y, x, equal_nan=True)
    runs = debugger.take_exported_runs()
    assert len(runs) == 1
    records = runs[0]["decoded"]["records"]
    expected_nan = 8 if all_nonfinite else 2
    counts = [
        r for r in records if r["record_kind"] == "SUMMARY_COUNT_BUNDLE_U64"
        and r["nan_count"] == expected_nan
    ]
    assert counts, records  # Shared load/store values may reuse one capture.
    finite = x.float().cpu()[torch.isfinite(x.cpu())]
    for record in counts:
        assert record["element_count"] == 16
        assert record["inf_count"] == (8 if all_nonfinite else 4)
        assert record["zero_count"] == (0 if all_nonfinite else 4)
        values = [
            v for v in records
            if v["record_kind"] == "SUMMARY_VALUE_BUNDLE_F32"
            and v["op_id"] == record["op_id"]
        ]
        assert len(values) == 1
        expected = dict(
            mean=finite.mean().item(),
            min=finite.min().item(),
            max=finite.max().item(),
            l2_norm=finite.norm().item()) if len(finite) else dict.fromkeys(
                ("mean", "min", "max", "l2_norm"), 0.0)
        for key, value in expected.items():
            assert values[0][key] == pytest.approx(value, abs=1e-5)


def test_hcu_overflow_and_recovery():
    x = torch.arange(2048, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    for capacity in (1, 1024):
        debugger.activate(level=1,
                          timeline=False,
                          record_capacity=capacity,
                          output_dir=None)
        hcu_summary_copy[(128, )](x, y, num_warps=1)
        torch.testing.assert_close(y, x)
        runs = debugger.take_exported_runs()
        assert len(runs) == 1
        decoded = runs[0]["decoded"]
        effective_capacity = decoded["header"]["capacity"]
        assert effective_capacity >= capacity
        assert len(decoded["records"]) <= effective_capacity
        if capacity == 1:
            assert decoded["header"]["overflow_count"] > 0
        else:
            assert decoded["header"]["overflow_count"] == 0
            assert len(decoded["records"]) > 1
        debugger.deactivate()


def test_hcu_debugger_toggle_preserves_plain_launch():

    @triton.jit
    def plain_copy(X, Y):
        i = tl.arange(0, 16)
        tl.store(Y + i, tl.load(X + i) + 1)

    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    for enabled in (False, True, False, True, False):
        if enabled:
            debugger.activate(auto_collect=True,
                              level=1,
                              timeline=False,
                              output_dir=None)
        else:
            debugger.deactivate()
        compiled = plain_copy[(1, )](x, y, num_warps=1)
        torch.testing.assert_close(y, x + 1)
        assert bool(getattr(compiled.metadata, "debug_enabled",
                            False)) == enabled
        assert bool(
            getattr(compiled.metadata, "debug_launch_hidden_arg",
                    False)) == enabled
        assert len(debugger.take_exported_runs()) == int(enabled)


# FlagPrism end


def test_hcu_block_pointer_capture():

    @triton.jit
    def block_copy(X, Y, N: tl.constexpr):
        ftl.debug_collect_start(level=1)
        x = tl.make_block_ptr(X, (N, ), (1, ), (0, ), (64, ), (0, ))
        y = tl.make_block_ptr(Y, (N, ), (1, ), (0, ), (64, ), (0, ))
        value = tl.load(x, boundary_check=(0, ), padding_option="zero")
        tl.store(y, value + 1, boundary_check=(0, ))
        ftl.debug_collect_end()

    x = torch.arange(37, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    debugger.activate(level=1, timeline=False, output_dir=None)
    block_copy[(1, )](x, y, 37, num_warps=1)
    torch.testing.assert_close(y, x + 1)
    runs = debugger.take_exported_runs()
    assert len(runs) == 1
    assert runs[0]["decoded"]["records"]
    assert runs[0]["decoded"]["header"]["overflow_count"] == 0
