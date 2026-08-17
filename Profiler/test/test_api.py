"""
Test module for FlagTree Profiler's API functionality.
No GPU kernel should be declared in this test.
Profile correctness tests involving GPU kernels should be placed in `test_profile.py`.
"""

import json
import flagtree.profiler as profiler
import pathlib
import pytest
import triton
from flagtree.profiler.hooks.hook import HookManager
from flagtree.profiler.hooks.launch import LaunchHook
from flagtree.profiler.hooks.instrumentation import InstrumentationHook


def _uses_cann_runtime():
    try:
        backend = triton.runtime.driver.active.get_current_target().backend
    except RuntimeError:
        return False
    return backend in {"ascend", "npu"}


def test_profile_single_session(tmp_path: pathlib.Path):
    temp_file0 = tmp_path / "test_profile0.hatchet"
    session_id0 = profiler.start(str(temp_file0.with_suffix("")))
    profiler.activate()
    profiler.deactivate()
    profiler.finalize()
    assert session_id0 == 0
    assert temp_file0.exists()

    temp_file1 = tmp_path / "test_profile1.hatchet"
    session_id1 = profiler.start(str(temp_file1.with_suffix("")))
    profiler.activate(session_id1)
    profiler.deactivate(session_id1)
    profiler.finalize(session_id1)
    assert session_id1 == session_id0 + 1
    assert temp_file1.exists()

    session_id2 = profiler.start("test")
    profiler.activate(session_id2)
    profiler.deactivate(session_id2)
    profiler.finalize()
    assert session_id2 == session_id1 + 1
    assert pathlib.Path("test.hatchet").exists()
    pathlib.Path("test.hatchet").unlink()


@pytest.mark.skipif(_uses_cann_runtime(),
                    reason="CANN sessions cannot overlap")
def test_profile_multiple_sessions(tmp_path: pathlib.Path):
    temp_file0 = tmp_path / "test_profile0.hatchet"
    profiler.start(str(temp_file0.with_suffix("")))
    temp_file1 = tmp_path / "test_profile1.hatchet"
    profiler.start(str(temp_file1.with_suffix("")))
    profiler.activate()
    profiler.deactivate()
    profiler.finalize()
    assert temp_file0.exists()
    assert temp_file1.exists()

    temp_file2 = tmp_path / "test_profile2.hatchet"
    session_id2 = profiler.start(str(temp_file2.with_suffix("")))
    temp_file3 = tmp_path / "test_profile3.hatchet"
    session_id3 = profiler.start(str(temp_file3.with_suffix("")))
    profiler.deactivate(session_id2)
    profiler.deactivate(session_id3)
    profiler.finalize()
    assert temp_file2.exists()
    assert temp_file3.exists()


def test_profile_decorator(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_profile_decorator.hatchet"

    @profiler.profile(name=str(temp_file.with_suffix("")))
    def foo0(a, b):
        return a + b

    foo0(1, 2)
    profiler.finalize()
    assert temp_file.exists()

    @profiler.profile
    def foo1(a, b):
        return a + b

    foo1(1, 2)
    profiler.finalize()
    default_file = pathlib.Path(profiler.DEFAULT_PROFILE_NAME + ".hatchet")
    assert default_file.exists()
    default_file.unlink()


def test_scope(tmp_path: pathlib.Path):
    # Scope can be annotated even when profiling is off
    with profiler.scope("test"):
        pass

    temp_file = tmp_path / "test_scope.hatchet"
    profiler.start(str(temp_file.with_suffix("")))
    with profiler.scope("test"):
        pass

    @profiler.scope("test")
    def foo():
        pass

    foo()

    profiler.enter_scope("test")
    profiler.exit_scope()

    profiler.enter_scope("test0")
    profiler.exit_scope("test0")

    profiler.finalize()
    assert temp_file.exists()


def test_hook(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_hook.hatchet"
    session_id0 = profiler.start(str(temp_file.with_suffix("")), hook="triton")
    profiler.activate(session_id0)
    profiler.activate(session_id0)
    assert len(HookManager.active_hooks) == 1, (
        "Activate a session multiple times should maintain a single instance of hook"
    )
    assert list(HookManager.session_hooks[session_id0].values())[0] is True
    profiler.deactivate(session_id0)
    assert list(HookManager.session_hooks[session_id0].values())[0] is False
    assert len(HookManager.active_hooks) == 0
    # Deactivate a session multiple times should not raise an error
    profiler.deactivate(session_id0)
    profiler.finalize(None)
    assert temp_file.exists()


@pytest.mark.skipif(_uses_cann_runtime(),
                    reason="native instrumentation hooks require CUDA or HIP")
def test_hook_manager(tmp_path: pathlib.Path):
    # Launch hook is a singleton
    HookManager.register(LaunchHook(), 0)
    HookManager.register(LaunchHook(), 0)
    assert len(HookManager.active_hooks) == 1
    assert isinstance(HookManager.active_hooks[0], LaunchHook)
    assert HookManager.session_hooks[0][HookManager.active_hooks[0]] is True

    # Only unregister one session
    HookManager.register(LaunchHook(), 1)
    HookManager.unregister(0)
    assert len(HookManager.active_hooks) == 1
    HookManager.unregister(1)
    assert len(HookManager.active_hooks) == 0

    # Heterogenous hooks
    HookManager.register(InstrumentationHook(""), 2)
    HookManager.register(LaunchHook(), 2)
    assert len(HookManager.active_hooks) == 2
    # Launch hook has a higher priority
    assert isinstance(HookManager.active_hooks[0], LaunchHook)
    assert isinstance(HookManager.active_hooks[1], InstrumentationHook)
    assert HookManager.session_hooks[2][HookManager.active_hooks[0]] is True
    assert HookManager.session_hooks[2][HookManager.active_hooks[1]] is True
    HookManager.unregister()
    assert len(HookManager.active_hooks) == 0


def test_scope_metrics(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_metrics.hatchet"
    session_id = profiler.start(str(temp_file.with_suffix("")))
    # Test different scope creation methods
    with profiler.scope("test0", {"a": 1.0}):
        pass

    @profiler.scope("test1", {"a": 1.0})
    def foo():
        pass

    foo()

    # After deactivation, the metrics should be ignored
    profiler.deactivate(session_id)
    profiler.enter_scope("test2", metrics={"a": 1.0})
    profiler.exit_scope()

    # Metrics should be recorded again after reactivation
    profiler.activate(session_id)
    profiler.enter_scope("test3", metrics={"a": 1.0})
    profiler.exit_scope()

    profiler.enter_scope("test3", metrics={"a": 1.0})
    profiler.exit_scope()

    # exit_scope can also take metrics
    profiler.enter_scope("test4")
    profiler.exit_scope(metrics={"b": 1.0})

    profiler.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 4
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test3":
            assert child["metrics"]["a"] == 2.0
        elif child["frame"]["name"] == "test4":
            assert child["metrics"]["b"] == 1.0


def test_scope_properties(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_properties.hatchet"
    profiler.start(str(temp_file.with_suffix("")))
    # Test different scope creation methods
    # Different from metrics, properties could be str
    with profiler.scope("test0", {"a (pty)": "1"}):
        pass

    @profiler.scope("test1", {"a (pty)": "1"})
    def foo():
        pass

    foo()

    # Properties do not aggregate
    profiler.enter_scope("test2", metrics={"a (pty)": 1.0})
    profiler.exit_scope()

    profiler.enter_scope("test2", metrics={"a (pty)": 1.0})
    profiler.exit_scope()

    profiler.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test2":
            assert child["metrics"]["a"] == 1.0
        elif child["frame"]["name"] == "test0":
            assert child["metrics"]["a"] == "1"


def test_scope_exclusive(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_exclusive.hatchet"
    profiler.start(str(temp_file.with_suffix("")))
    # metric a only appears in the outermost scope
    # metric b only appears in the innermost scope
    # both metrics do not appear in the root scope
    with profiler.scope("test0", metrics={"a (exc)": "1"}):
        with profiler.scope("test1", metrics={"b (exc)": "1"}):
            pass

    profiler.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    root_metrics = data[0]["metrics"]
    assert len(root_metrics) == 0
    test0_frame = data[0]["children"][0]
    test0_metrics = test0_frame["metrics"]
    assert len(test0_metrics) == 1
    assert test0_metrics["a"] == "1"
    test1_frame = test0_frame["children"][0]
    test1_metrics = test1_frame["metrics"]
    assert len(test1_metrics) == 1
    assert test1_metrics["b"] == "1"


def test_state(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_state.hatchet"
    profiler.start(str(temp_file.with_suffix("")))
    profiler.enter_scope("test0")
    profiler.enter_state("state")
    profiler.enter_scope("test1", metrics={"a": 1.0})
    profiler.exit_scope()
    profiler.exit_state()
    profiler.exit_scope()
    profiler.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    # test0->test1->state
    assert len(data[0]["children"]) == 1
    child = data[0]["children"][0]
    assert child["frame"]["name"] == "test0"
    assert len(child["children"]) == 1
    child = child["children"][0]
    assert child["frame"]["name"] == "test1"
    assert len(child["children"]) == 1
    child = child["children"][0]
    assert child["frame"]["name"] == "state"
    assert child["metrics"]["a"] == 1.0


def test_context_depth(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_context_depth.hatchet"
    session_id = profiler.start(str(temp_file.with_suffix("")))
    assert profiler.context.depth(session_id) == 0
    profiler.enter_scope("test0")
    assert profiler.context.depth(session_id) == 1
    profiler.enter_scope("test1")
    assert profiler.context.depth(session_id) == 2
    profiler.exit_scope()
    assert profiler.context.depth(session_id) == 1
    profiler.exit_scope()
    assert profiler.context.depth(session_id) == 0
    profiler.finalize()


def test_throw(tmp_path: pathlib.Path):
    # Catch an exception thrown by c++
    session_id = 100
    temp_file = tmp_path / "test_throw.hatchet"
    activate_error = ""
    try:
        session_id = profiler.start(str(temp_file.with_suffix("")))
        profiler.activate(session_id + 1)
    except Exception as e:
        activate_error = str(e)
    finally:
        profiler.finalize()
    assert "Session has not been initialized: " + str(session_id +
                                                      1) in activate_error

    deactivate_error = ""
    try:
        session_id = profiler.start(str(temp_file.with_suffix("")))
        profiler.deactivate(session_id + 1)
    except Exception as e:
        deactivate_error = str(e)
    finally:
        profiler.finalize()
    assert "Session has not been initialized: " + str(session_id +
                                                      1) in deactivate_error
