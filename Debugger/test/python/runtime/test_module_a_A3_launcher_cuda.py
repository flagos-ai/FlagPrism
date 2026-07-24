# SPDX-License-Identifier: MIT
"""A-3: CUDA keeps its stock launcher ABI while Debugger CUDA support is deferred."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

pytest.importorskip("triton.backends.nvidia.driver")
from triton.backends.nvidia import driver as nvidia_driver


def _launcher_src():
    return SimpleNamespace(
        constants={},
        fn=SimpleNamespace(arg_names=["x"]),
        signature={0: "*fp32"},
    )


def _launcher_metadata():
    return SimpleNamespace(
        tensordesc_meta=None,
        cluster_dims=(1, 1, 1),
        global_scratch_size=0,
        global_scratch_align=1,
        profile_scratch_size=0,
        profile_scratch_align=1,
        launch_cooperative_grid=False,
        launch_pdl=False,
    )


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_a_A3_make_launcher_uses_standard_cuda_abi():
    source = nvidia_driver.make_launcher({}, {0: "*fp32"}, None)
    params = "void *params[] = { &arg0, &global_scratch, &profile_scratch };"
    assert "debug_hidden_arg" not in source
    assert params in source


@pytest.mark.module_a
@pytest.mark.module_a_a3
def test_module_a_A3_cuda_launcher_forwards_original_arguments(monkeypatch):
    launch_module = SimpleNamespace()
    launch_module.launch = lambda *args: setattr(launch_module, "seen", args)
    monkeypatch.setattr(
        nvidia_driver,
        "compile_module_from_src",
        lambda *args, **kwargs: launch_module,
    )
    monkeypatch.setattr(nvidia_driver, "library_dirs", lambda: [])

    launcher = nvidia_driver.CudaLauncher(_launcher_src(), _launcher_metadata())
    launcher(2, 1, 1, object(), object(), object(), {"grid": (2, 1, 1)}, None, None, 99)

    assert launch_module.seen[-1] == 99
    assert not hasattr(launcher, "debug_launch_hidden_arg")
