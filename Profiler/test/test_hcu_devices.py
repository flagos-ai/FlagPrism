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
def hcu_devices():
    if ".hcu." not in type(triton.runtime.driver.active).__module__:
        pytest.skip("HCU device required")
    previous = torch.cuda.current_device()
    yield
    profiler.finalize()
    torch.cuda.set_device(previous)


@pytest.mark.parametrize("hook", ["triton", "instrumentation"])
def test_multidevice_ir_attribution(tmp_path, hook):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two HCU devices required")
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
    session = profiler.start(str(path), backend="hcu", hook=hook)
    try:
        for device in (0, 1):
            with torch.cuda.device(device):
                device_copy[(1, )](*tensors[device])
    finally:
        profiler.finalize(session)
    report = json.loads(path.with_suffix(".vendor.json").read_text())
    for source in ("hcu_activity", "flagtree_ir_kernel", "flagtree_ir_op"):
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
