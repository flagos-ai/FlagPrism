# FlagPrism：real-device runtime activities, threaded multi-GPU and graph replay.
import ctypes
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
from flagtree import profiler


@triton.jit
def extended_probe(X, Y):
    i = tl.arange(0, 32)
    tl.store(Y + i, tl.load(X + i) + 1)


@pytest.fixture(autouse=True)
def metax_only():
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    previous = torch.cuda.current_device()
    yield
    profiler.finalize()
    torch.cuda.set_device(previous)


def document(path):
    return json.loads(path.with_suffix(".vendor.json").read_text())


def test_runtime_memory_and_api_activities(tmp_path):
    import os
    runtime = ctypes.CDLL(
        os.path.join(os.getenv("MACA_PATH", "/opt/maca"),
                     "lib/libmcruntime.so"))
    runtime.mcMemsetAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
    ]
    runtime.mcMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_void_p
    ]
    x = torch.empty(256, device="cuda")
    host = torch.arange(256, dtype=torch.float32, pin_memory=True)
    result = torch.empty_like(host, pin_memory=True)
    path = tmp_path / "runtime"
    session = profiler.start(str(path),
                             backend="metax",
                             mode="runtime_base:runtime_events=true")
    try:
        assert runtime.mcMemsetAsync(x.data_ptr(), 0, 1024, None) == 0
        profiler.deactivate(session)
        assert runtime.mcMemsetAsync(x.data_ptr(), 3, 512, None) == 0
        profiler.activate(session)
        assert runtime.mcMemcpyAsync(x.data_ptr(), host.data_ptr(), 1024, 1,
                                     None) == 0
        x.add_(1)
        assert runtime.mcMemcpyAsync(result.data_ptr(), x.data_ptr(), 1024, 2,
                                     None) == 0
        torch.cuda.synchronize()
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(result, host + 1)
    associations = document(path)["associations"]
    copies = [a for a in associations if a["source"] == "mcpti_memcpy"]
    sets = [a for a in associations if a["source"] == "mcpti_memset"]
    apis = [a for a in associations if a["source"] == "mcpti_runtime_api"]
    assert len(copies) == 2 and len(sets) == 1
    assert {a["metrics"]["metax.copy_kind"] for a in copies} == {1, 2}
    assert all(a["metrics"]["metax.bytes"] == 1024 for a in copies + sets)
    assert apis and all(a["metrics"]["metax.return_value"] == 0 for a in apis)
    assert any("Synchronize" in a["runtime_event"]["op_name"] for a in apis)
    timeline = json.loads(path.with_suffix(".timeline.json").read_text())
    categories = {
        e.get("cat")
        for e in timeline["traceEvents"] if e.get("ph") == "X"
    }
    assert "metax_runtime:mcpti_memcpy" in categories
    assert "metax_runtime:mcpti_memset" in categories
    assert "metax_runtime:mcpti_runtime_api" in categories
    tree = path.with_suffix(".hatchet").read_text()
    assert "mcMemcpyAsync" not in tree and "mcDeviceSynchronize" not in tree
    assert "metax_memcpy" not in tree and "metax_memset" not in tree


@pytest.mark.parametrize("repeat", range(3))
def test_threaded_multidevice_session(tmp_path, repeat):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two visible MetaX devices required")
    tensors = []
    for device in (0, 1):
        with torch.cuda.device(device):
            x = torch.ones(32, device=f"cuda:{device}")
            y = torch.empty_like(x)
            stream = torch.cuda.Stream()
            extended_probe[(1, )](x, y)
            torch.cuda.synchronize()
            tensors.append((x, y, stream))
    path = tmp_path / "threaded"
    torch.cuda.set_device(1)
    session = profiler.start(str(path), backend="metax")

    def launch(device):
        torch.cuda.set_device(device)
        x, y, stream = tensors[device]
        with torch.cuda.stream(stream):
            for _ in range(200):
                extended_probe[(1, )](x, y)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, (0, 1)))
    finally:
        profiler.finalize(session)
    assert torch.cuda.current_device() == 1
    kernels = [
        a for a in document(path)["associations"]
        if a["source"] == "mcpti_activity"
        and "extended_probe" in a["runtime_event"]["op_name"]
    ]
    assert len(kernels) == 400
    assert {a["runtime_event"]["device_id"] for a in kernels} == {0, 1}
    for device in (0, 1):
        assert sum(a["runtime_event"]["device_id"] == device
                   for a in kernels) == 200
        torch.testing.assert_close(tensors[device][1], tensors[device][0] + 1)


def test_graph_replay_activity(tmp_path):
    x = torch.ones(32, device="cuda")
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        extended_probe[(1, )](x, y)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        extended_probe[(1, )](x, y)
    path = tmp_path / "graph"
    session = profiler.start(str(path), backend="metax")
    try:
        for _ in range(3):
            graph.replay()
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(y, x + 1)
    kernels = [
        a for a in document(path)["associations"]
        if a["source"] == "mcpti_activity"
        and "extended_probe" in a["runtime_event"]["op_name"]
    ]
    assert len(kernels) == 3


# FlagPrism end


# FlagPrism：hardware counters cover active ranges and never imply per-kernel attribution.
@pytest.mark.parametrize("metric", [
    "dram_read_bytes", "dram_write_bytes", "dram_read_throughput",
    "dram_write_throughput", "inst_executed"
])
def test_hardware_counter_ranges(tmp_path, metric):
    x = torch.arange(65536, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    path = tmp_path / "counters"
    mode = f"runtime_base:vendor_metrics={metric}"
    session = profiler.start(str(path), backend="metax", mode=mode)
    try:
        for _ in range(20):
            torch.add(x, 1, out=y)
        profiler.deactivate(session)
        for _ in range(20):
            torch.add(x, 5, out=y)
        profiler.activate(session)
        for _ in range(20):
            torch.add(x, 1, out=y)
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(y, x + 1)
    report = document(path)
    ranges = [
        a for a in report["associations"]
        if a["source"] == "mcpti_hardware_counter"
    ]
    assert len(ranges) == 2, report["degrade_reasons"]
    for captured in ranges:
        metrics = captured["metrics"]
        value = metrics[f"metax.hardware.{metric}"]
        assert value > 0
        if metric.endswith("throughput"):
            duration = captured["runtime_event"]["end_time_ns"] - captured[
                "runtime_event"]["start_time_ns"]
            assert value == pytest.approx(
                metrics[f"metax.hardware.{metric}.bytes"] * 1e9 / duration)
        assert "not per-kernel" in captured["note"]
    kernels = [
        a for a in report["associations"] if a["source"] == "mcpti_activity"
    ]
    assert len(kernels) == 40


# FlagPrism end


# FlagPrism：validate that a hardware instruction delta scales with actual work.
def test_instruction_counter_workload_scaling(tmp_path):
    x = torch.ones(65536, device="cuda")
    y = torch.empty_like(x)
    torch.add(x, 1, out=y)
    torch.cuda.synchronize()
    measured = []
    for launches in (20, 40):
        path = tmp_path / f"instructions_{launches}"
        session = profiler.start(
            str(path),
            backend="metax",
            mode="runtime_base:vendor_metrics=inst_executed")
        try:
            for _ in range(launches):
                torch.add(x, 1, out=y)
        finally:
            profiler.finalize(session)
        report = document(path)
        ranges = [
            a for a in report["associations"]
            if a["source"] == "mcpti_hardware_counter"
        ]
        assert len(ranges) == 1, report["degrade_reasons"]
        measured.append(ranges[0]["metrics"]["metax.hardware.inst_executed"])
    assert measured[0] > 0
    assert measured[1] == pytest.approx(2 * measured[0], rel=0.05)


# FlagPrism end
