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
def hcu_only():
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU device required")
    previous = torch.cuda.current_device()
    yield
    profiler.finalize()
    torch.cuda.set_device(previous)


def document(path):
    return json.loads(path.with_suffix(".vendor.json").read_text())


@pytest.mark.parametrize("repeat", range(3))
def test_threaded_multidevice_session(tmp_path, repeat):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two visible HCU devices required")
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
    session = profiler.start(str(path), backend="hcu")

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
        if a["source"] == "hcu_activity"
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
    session = profiler.start(str(path), backend="hcu")
    try:
        for _ in range(3):
            graph.replay()
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(y, x + 1)
    kernels = [
        a for a in document(path)["associations"]
        if a["source"] == "hcu_activity"
        and "extended_probe" in a["runtime_event"]["op_name"]
    ]
    assert len(kernels) == 3


# FlagPrism end


def test_runtime_memory_and_api_activities(tmp_path):
    runtime = ctypes.CDLL("/opt/dtk/lib/libamdhip64.so")
    runtime.hipMemsetAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p
    ]
    runtime.hipMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_void_p
    ]
    x = torch.empty(256, device="cuda")
    host = torch.arange(256, dtype=torch.float32, pin_memory=True)
    result = torch.empty_like(host, pin_memory=True)
    path = tmp_path / "runtime"
    session = profiler.start(str(path),
                             backend="hcu",
                             mode="runtime_base:runtime_events=true")
    try:
        assert runtime.hipMemsetAsync(x.data_ptr(), 0, 1024, None) == 0
        profiler.deactivate(session)
        assert runtime.hipMemsetAsync(x.data_ptr(), 3, 512, None) == 0
        profiler.activate(session)
        assert runtime.hipMemcpyAsync(x.data_ptr(), host.data_ptr(), 1024, 1,
                                      None) == 0
        x.add_(1)
        assert runtime.hipMemcpyAsync(result.data_ptr(), x.data_ptr(), 1024, 2,
                                      None) == 0
        torch.cuda.synchronize()
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(result, host + 1)
    activities = document(path)["associations"]
    copies = [
        a for a in activities
        if a["metrics"].get("hcu.activity_kind") == "memcpy"
    ]
    apis = [
        a for a in activities
        if a["metrics"].get("hcu.activity_kind") == "host_api"
    ]
    sets = [
        a for a in activities
        if a["metrics"].get("hcu.activity_kind") == "memset"
    ]
    assert len(sets) == 1 and sets[0]["metrics"]["hcu.bytes"] == 1024
    assert len(copies) == 2
    assert all(a["metrics"]["hcu.bytes"] == 1024 for a in copies)
    assert apis and any("Synchronize" in a["runtime_event"]["op_name"]
                        for a in apis)
    tree = path.with_suffix(".hatchet").read_text()
    assert "hipMemcpyAsync" not in tree and "hipDeviceSynchronize" not in tree


@pytest.mark.parametrize(
    "mode", ["runtime_events=invalid", "device_id=-1", "device_id=9999"])
def test_invalid_configuration_is_retryable(tmp_path, mode):
    with pytest.raises((RuntimeError, ValueError)):
        profiler.start(str(tmp_path / "invalid"), backend="hcu", mode=mode)
    session = profiler.start(str(tmp_path / "retry"), backend="hcu")
    profiler.finalize(session)


def test_automatic_backend(tmp_path):
    from flagtree.profiler.profile import _select_backend
    assert _select_backend() == "hcu"


@pytest.mark.parametrize("first,second", [("hcu", "roctracer"),
                                          ("roctracer", "hcu")])
def test_legacy_collector_overlap_rejected(tmp_path, first, second):
    session = profiler.start(str(tmp_path / "first"), backend=first)
    try:
        with pytest.raises(RuntimeError, match="overlapping"):
            profiler.start(str(tmp_path / "second"), backend=second)
    finally:
        profiler.finalize(session)
    session = profiler.start(str(tmp_path / "retry"), backend=second)
    profiler.finalize(session)


def test_required_unsupported_metric_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="unsupported"):
        profiler.start(
            str(tmp_path / "required"),
            backend="hcu",
            mode="vendor_metrics=unsupported_counter:require_metrics=true")
    session = profiler.start(str(tmp_path / "retry"), backend="hcu")
    profiler.finalize(session)
