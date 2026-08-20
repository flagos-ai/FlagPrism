import pathlib
import pytest

from flagtree.profiler.native import runtime_binding
from flagtree.profiler.profile import _select_backend

profiler_native = runtime_binding()


def test_record():
    id0 = profiler_native.record_scope()
    id1 = profiler_native.record_scope()
    assert id1 == id0 + 1


def test_state():
    profiler_native.enter_state("zero")
    profiler_native.exit_state()


def test_scope():
    id0 = profiler_native.record_scope()
    profiler_native.enter_scope(id0, "zero")
    id1 = profiler_native.record_scope()
    profiler_native.enter_scope(id1, "one")
    profiler_native.exit_scope(id1, "one")
    profiler_native.exit_scope(id0, "zero")


def test_op():
    id0 = profiler_native.record_scope()
    profiler_native.enter_op(id0, "zero")
    profiler_native.exit_op(id0, "zero")


@pytest.mark.parametrize("source", ["shadow", "python"])
def test_context(source: str, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_context.hatchet"
    session_id = profiler_native.start(str(temp_file.with_suffix("")), source,
                                       "tree", _select_backend(), "")
    depth = profiler_native.get_context_depth(session_id)
    profiler_native.finalize(session_id, "hatchet")
    assert depth >= 0
    assert temp_file.exists()


def test_session(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_session.hatchet"
    session_id = profiler_native.start(str(temp_file.with_suffix("")),
                                       "shadow", "tree", _select_backend())
    profiler_native.deactivate(session_id)
    profiler_native.activate(session_id)
    profiler_native.finalize(session_id, "hatchet")
    profiler_native.finalize_all("hatchet")
    assert temp_file.exists()


def test_add_metrics(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_add_metrics.hatchet"
    profiler_native.start(str(temp_file.with_suffix("")), "shadow", "tree",
                          _select_backend())
    id1 = profiler_native.record_scope()
    profiler_native.enter_scope(id1, "one")
    profiler_native.add_metrics(id1, {"a": 1.0, "b": 2.0})
    profiler_native.exit_scope(id1, "one")
    profiler_native.finalize_all("hatchet")
    assert temp_file.exists()
