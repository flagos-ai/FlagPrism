# FlagPrism：device selection and IR attribution must agree with actual execution.
import json

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
from flagtree import profiler


@triton.jit
def device_copy(X, Y):
    i = tl.arange(0, 32)
    tl.store(Y + i, tl.load(X + i) + 1)


@pytest.fixture(autouse=True)
def metax_devices():
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    previous = torch.cuda.current_device()
    yield
    profiler.finalize()
    torch.cuda.set_device(previous)


@pytest.mark.parametrize("target", [0, 1])
def test_explicit_counter_device_and_restore(tmp_path, target):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two MetaX devices required")
    caller = 1 - target
    with torch.cuda.device(target):
        x = torch.ones(65536, device="cuda")
        y = torch.empty_like(x)
        torch.add(x, 1, out=y)
        torch.cuda.synchronize()
    torch.cuda.set_device(caller)
    path = tmp_path / "explicit_counter"
    session = profiler.start(
        str(path),
        backend="metax",
        mode=f"runtime_base:device_id={target}:vendor_metrics=inst_executed")
    assert torch.cuda.current_device() == caller
    try:
        for _ in range(2):
            with torch.cuda.device(target):
                for _ in range(20):
                    torch.add(x, 1, out=y)
            profiler.deactivate(session)
            assert torch.cuda.current_device() == caller
            profiler.activate(session)
            assert torch.cuda.current_device() == caller
    finally:
        profiler.finalize(session)
    assert torch.cuda.current_device() == caller
    torch.testing.assert_close(y, x + 1)
    report = json.loads(path.with_suffix(".vendor.json").read_text())
    counters = [
        a for a in report["associations"]
        if a["source"] == "mcpti_hardware_counter"
    ]
    assert counters, report["degrade_reasons"]
    assert all(a["runtime_event"]["device_id"] == target for a in counters)
    # Finalize after the last activate can add an empty range; test busy ranges.
    assert sum(a["metrics"]["metax.hardware.inst_executed"] > 0
               for a in counters) >= 2
    kernels = [
        a for a in report["associations"] if a["source"] == "mcpti_activity"
    ]
    assert len(kernels) == 40
    assert {a["runtime_event"]["device_id"] for a in kernels} == {target}


@pytest.mark.parametrize("hook", ["triton", "instrumentation"])
def test_multidevice_ir_attribution(tmp_path, hook):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two MetaX devices required")
    tensors = []
    for device in (0, 1):
        with torch.cuda.device(device):
            x = torch.ones(32, device="cuda")
            y = torch.empty_like(x)
            device_copy[(1, )](x, y)
            torch.cuda.synchronize()
            tensors.append((x, y))
    torch.cuda.set_device(0)
    path = tmp_path / hook
    session = profiler.start(str(path), backend="metax", hook=hook)
    try:
        for device in (0, 1):
            with torch.cuda.device(device):
                device_copy[(1, )](*tensors[device])
    finally:
        profiler.finalize(session)
    report = json.loads(path.with_suffix(".vendor.json").read_text())
    for source in ("mcpti_activity", "flagtree_ir_kernel", "flagtree_ir_op"):
        records = [a for a in report["associations"] if a["source"] == source]
        assert records, source
        assert {a["runtime_event"]["device_id"] for a in records} == {0, 1}
    ir_kernels = [
        a for a in report["associations"]
        if a["source"] == "flagtree_ir_kernel"
    ]
    assert len(ir_kernels) == 2
    assert [a["runtime_event"]["device_id"] for a in ir_kernels] == [0, 1]
    for association in report["associations"]:
        if association["source"] in ("flagtree_ir_kernel", "flagtree_ir_op"):
            event = association["runtime_event"]
            assert event["device_id"] == event["correlation_id"]
    tree = json.loads(path.with_suffix(".hatchet").read_text())[0]

    def walk(node):
        yield node
        for child in node.get("children", []):
            yield from walk(child)

    device_nodes = [
        n for n in walk(tree)
        if n.get("frame", {}).get("name", "").startswith("IR device ")
    ]
    assert {n["metrics"]["device_id"] for n in device_nodes} == {0, 1}
    for node in device_nodes:
        assert node["children"]
        assert all(c["metrics"]["device_id"] == node["metrics"]["device_id"]
                   for c in node["children"])
    for x, y in tensors:
        torch.testing.assert_close(y, x + 1)


@pytest.mark.parametrize(
    "selection", ["-1", "bad", "1junk", "999999", "184467440737095516160"])
def test_invalid_device_rejected_and_retry(tmp_path, selection):
    with pytest.raises((ValueError, RuntimeError), match="device_id"):
        profiler.start(
            str(tmp_path / "invalid"),
            backend="metax",
            mode=
            f"runtime_base:device_id={selection}:vendor_metrics=inst_executed")
    session = profiler.start(str(tmp_path / "retry"), backend="metax")
    profiler.finalize(session)


# FlagPrism end
