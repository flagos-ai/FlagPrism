from types import SimpleNamespace

from flagtree.debugger import native


def test_compiler_binding_uses_libtriton_plugin(monkeypatch):
    compiler = object()
    libtriton = SimpleNamespace(debugger=compiler)
    monkeypatch.setattr(
        native,
        "_optional_module",
        lambda name: libtriton if name == "triton._C.libtriton" else None,
    )

    assert native.compiler_binding() is compiler


def test_runtime_binding_uses_standalone_extension(monkeypatch):
    runtime = object()
    monkeypatch.setattr(
        native,
        "_optional_module",
        lambda name: runtime if name == "flagtree.debugger._native" else None,
    )

    assert native.runtime_binding() is runtime
