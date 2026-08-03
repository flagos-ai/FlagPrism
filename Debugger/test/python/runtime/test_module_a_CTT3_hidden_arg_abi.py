# SPDX-License-Identifier: MIT
"""CTT-3: Ascend appends the Debugger hidden argument after user arguments."""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

pytest.importorskip("triton.backends.ascend.driver")
from triton.backends.ascend.driver import NPULauncher


def _launcher():
    launcher = object.__new__(NPULauncher)
    launcher.compile_only = False
    launcher.enable_msprof_register_tensor = False
    launcher.metadata = SimpleNamespace(
        debug_enabled=True,
        debug_launch_hidden_arg=True,
    )
    launcher.launch = lambda *args, **kwargs: setattr(launcher, "seen", args) or 0
    return launcher


@pytest.mark.module_a
@pytest.mark.module_a_ctt3
def test_module_a_CTT3_hidden_arg_is_last_launch_tuple_element(monkeypatch):
    from flagtree.debugger import api

    @contextmanager
    def launch_context(*args, **kwargs):
        del args, kwargs
        yield (0x11223344,)

    monkeypatch.setattr(api, "ascend_launch_context", launch_context)
    launcher = _launcher()
    launcher(1, 1, 1, 0, 0x1234, {"hash": "unit"}, None, None, None, 99)

    assert launcher.seen[-2:] == (99, 0x11223344)


@pytest.mark.module_a
@pytest.mark.module_a_ctt3
def test_module_a_CTT3_rejects_hidden_arg_count_mismatch(monkeypatch):
    from flagtree.debugger import api

    monkeypatch.setattr(
        api,
        "prepare_kernel_launch",
        lambda *args, **kwargs: api.PreparedKernelLaunch(kernel_args=()),
    )
    metadata = SimpleNamespace(
        debug_enabled=True,
        debug_launch_hidden_arg=True,
    )
    with pytest.raises(RuntimeError, match="requires 1 debugger hidden argument"):
        with api.ascend_launch_context(metadata, (1, 1, 1), 0):
            pass
