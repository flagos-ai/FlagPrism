# FlagPrism：experimental local clock probe, not a globally aligned device timeline.
import json
import re
import statistics

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
from triton.language import core


@core.extern
def metax_clock(_semantic=None):
    return core.extern_elementwise(
        "",
        "", [], {(): ("llvm.readcyclecounter", core.uint64)},
        is_pure=False,
        _semantic=_semantic)


@triton.jit
def clock_interval(X, Y, TIMES, N: tl.constexpr):
    offset = tl.program_id(0) * 32 + tl.arange(0, 32)
    begin = metax_clock()
    value = tl.load(X + offset)
    for _ in range(N):
        value = value * 1.00001 + 0.00001
    tl.store(Y + offset, value)
    end = metax_clock()
    tl.store(TIMES + tl.program_id(0) * 2, begin)
    tl.store(TIMES + tl.program_id(0) * 2 + 1, end)


@pytest.mark.parametrize("programs", [1, 8])
def test_local_clock_interval_probe(tmp_path, programs):
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX device required")
    x = torch.ones(programs * 32, device="cuda")
    y = torch.empty_like(x)
    times = torch.empty(programs * 2, device="cuda", dtype=torch.int64)
    measurements = {}
    for iterations in (0, 1000, 10000):
        compiled = clock_interval[(programs, )](x, y, times, iterations)
        samples = []
        for _ in range(5):
            clock_interval[(programs, )](x, y, times, iterations)
            torch.cuda.synchronize()
            captured = times.cpu().reshape(programs, 2)
            assert torch.all(captured[:, 0] > 0)
            assert torch.all(captured[:, 1] >= captured[:, 0])
            samples.extend((captured[:, 1] - captured[:, 0]).tolist())
        assert torch.isfinite(y).all()
        assert torch.all(y >= x)
        measurements[str(iterations)] = statistics.median(samples)
        llvm = compiled.asm["llir"]
        reads = list(
            re.finditer(r"call (?:noundef )?i64 @llvm\.readcyclecounter\(\)",
                        llvm))
        assert len(reads) == 2
        if iterations:
            # A retained dependency loop/FMA must lie between the two reads.
            interval = llvm[reads[0].end():reads[1].start()]
            assert "@llvm.fma.f32" in interval and "phi" in interval
        (tmp_path / f"clock_{iterations}.ll").write_text(llvm)
    assert measurements["1000"] > measurements["0"]
    assert measurements["10000"] > 4 * measurements["1000"]
    (tmp_path / "clock-probe.json").write_text(
        json.dumps(
            {
                "programs": programs,
                "median_local_counter_ticks": measurements,
                "source": "llvm.readcyclecounter (SDK clock64 intrinsic)",
                "cross_sm_clock_alignment_verified": False,
                "constant_tick_to_ns_conversion_verified": False,
                "production_internal_timeline_enabled": False,
            },
            indent=2))


# FlagPrism end
